import Foundation
import Metal
import Darwin

private let maximumHeadDim = 128
private let threadgroupWidth = 128

private struct PrefixTile {
    var absoluteStart: UInt32
    var validTokens: UInt32
    var keyOffsetElements: UInt32
    var valueOffsetElements: UInt32
}

private struct QueryGroup {
    var rowOffset: UInt32
    var rowCount: UInt32
    var queryHead: UInt32
    var kvHead: UInt32
    var tileOffset: UInt32
    var tileCount: UInt32
}

private struct Parameters {
    var queryRows: UInt32
    var queryHeads: UInt32
    var kvHeads: UInt32
    var tokens: UInt32
    var headDim: UInt32
    var tileCapacityTokens: UInt32
    var kvStride: UInt32
    var sharedPrefix: UInt32
}

struct CaseSpec {
    let name: String
    let tokens: UInt32
    let queryRows: UInt32
    let queryHeads: UInt32
    let kvHeads: UInt32
    let headDim: UInt32
    let tileTokens: UInt32
    let groupRows: UInt32
    let seed: UInt64
    let sharedPrefix: Bool
}

private struct Problem {
    let spec: CaseSpec
    var queries: [Float]
    var keys: [Float16]
    var values: [Float16]
    var tiles: [PrefixTile]
    var scheduleA: [QueryGroup]
    var scheduleB: [QueryGroup]
    var scheduleTilesA: [PrefixTile]
    var scheduleTilesB: [PrefixTile]
    var rowIdsA: [UInt32]
    var rowIdsB: [UInt32]
}

private enum Path: String, CaseIterable, Hashable {
    case perRow = "per_row"
    case fixedTilePerRow = "fixed_tile_per_row"
    case sharedReadUnconstrained = "shared_read_unconstrained"
    case sharedReadFixedReduction = "shared_read_fixed_reduction"
}

private struct RunResult {
    let output: [Float]
    let samplesMS: [Double]
    let medianMS: Double
    let perBlockSharedBytes: UInt64
    let estimatedKVElementsRead: UInt64
    let estimatedTileLoads: UInt64
    let outputElementsWritten: UInt64
}

private struct SplitMix64 {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var value = state
        value = (value ^ (value >> 30)) &* 0xbf58476d1ce4e5b9
        value = (value ^ (value >> 27)) &* 0x94d049bb133111eb
        return value ^ (value >> 31)
    }

    mutating func unitFloat() -> Float {
        let bits = next() >> 40
        return Float(bits) / 16_777_215.0 * 2.0 - 1.0
    }
}

private func tileCount(_ spec: CaseSpec) -> UInt32 {
    spec.tokens / spec.tileTokens + (spec.tokens % spec.tileTokens == 0 ? 0 : 1)
}

private func effectiveGroupRows(_ spec: CaseSpec) -> UInt32 {
    spec.sharedPrefix ? spec.groupRows : 1
}

private func tileFor(_ spec: CaseSpec, _ index: UInt32) -> PrefixTile {
    let start = index * spec.tileTokens
    return PrefixTile(absoluteStart: start,
                      validTokens: min(spec.tileTokens, spec.tokens - start),
                      keyOffsetElements: start * spec.headDim,
                      valueOffsetElements: start * spec.headDim)
}

private func appendSchedule(_ spec: CaseSpec, _ base: [PrefixTile], _ variantB: Bool,
                            _ groups: inout [QueryGroup], _ rowIds: inout [UInt32],
                            _ scheduleTiles: inout [PrefixTile]) {
    let rowsPerGroup = effectiveGroupRows(spec)
    var rows = Array(0..<spec.queryRows)
    if variantB {
        rows = Array(stride(from: UInt32(0), to: spec.queryRows, by: 2)) +
            Array(stride(from: UInt32(1), to: spec.queryRows, by: 2))
    }
    for queryHead in 0..<spec.queryHeads {
        var start: UInt32 = 0
        while start < spec.queryRows {
            let count = min(rowsPerGroup, spec.queryRows - start)
            let group = QueryGroup(
                rowOffset: UInt32(rowIds.count), rowCount: count,
                queryHead: queryHead,
                kvHead: queryHead * spec.kvHeads / spec.queryHeads,
                tileOffset: UInt32(scheduleTiles.count),
                tileCount: UInt32(base.count))
            groups.append(group)
            for offset in 0..<count {
                rowIds.append(rows[Int(start + offset)])
            }
            let groupIndex = groups.count - 1
            if !variantB && groupIndex % 2 == 1 {
                scheduleTiles.append(contentsOf: base.reversed())
            } else if variantB && groupIndex % 2 == 1 {
                let shift = groupIndex % base.count
                scheduleTiles.append(contentsOf: (0..<base.count).map {
                    base[(shift + $0) % base.count]
                })
            } else {
                scheduleTiles.append(contentsOf: base)
            }
            start += count
        }
    }
}

private func makeProblem(_ spec: CaseSpec) throws -> Problem {
    guard spec.tokens > 0, spec.queryRows > 0, spec.queryHeads > 0,
          spec.kvHeads > 0, spec.headDim > 0, spec.tileTokens > 0,
          spec.groupRows > 0, spec.queryRows <= spec.tokens,
          spec.queryHeads % spec.kvHeads == 0,
          spec.headDim <= maximumHeadDim else {
        throw NSError(domain: "prefix_attention", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "invalid case dimensions"])
    }
    var generator = SplitMix64(state: spec.seed)
    let queryCount = Int(spec.queryRows * spec.queryHeads * spec.headDim)
    var queries = [Float](repeating: 0, count: queryCount)
    for index in queries.indices {
        queries[index] = generator.unitFloat()
    }
    let keyRows: UInt32 = spec.sharedPrefix ? 1 : spec.queryRows
    let kvCount = Int(keyRows * spec.kvHeads * spec.tokens * spec.headDim)
    var keys = [Float16](repeating: 0, count: kvCount)
    var values = [Float16](repeating: 0, count: kvCount)
    for index in keys.indices {
        keys[index] = Float16(generator.unitFloat())
        values[index] = Float16(generator.unitFloat())
    }
    let tiles = (0..<tileCount(spec)).map { tileFor(spec, $0) }
    var scheduleA: [QueryGroup] = []
    var scheduleB: [QueryGroup] = []
    var scheduleTilesA: [PrefixTile] = []
    var scheduleTilesB: [PrefixTile] = []
    var rowIdsA: [UInt32] = []
    var rowIdsB: [UInt32] = []
    appendSchedule(spec, tiles, false, &scheduleA, &rowIdsA, &scheduleTilesA)
    appendSchedule(spec, tiles, true, &scheduleB, &rowIdsB, &scheduleTilesB)
    let problem = Problem(spec: spec, queries: queries, keys: keys, values: values,
                          tiles: tiles, scheduleA: scheduleA, scheduleB: scheduleB,
                          scheduleTilesA: scheduleTilesA,
                          scheduleTilesB: scheduleTilesB, rowIdsA: rowIdsA,
                          rowIdsB: rowIdsB)
    try validate(problem)
    return problem
}

private func validate(_ problem: Problem) throws {
    let spec = problem.spec
    let expectedQuery = Int(spec.queryRows) * Int(spec.queryHeads) * Int(spec.headDim)
    let keyRows: UInt32 = spec.sharedPrefix ? 1 : spec.queryRows
    let expectedKV = Int(keyRows) * Int(spec.kvHeads) * Int(spec.tokens) * Int(spec.headDim)
    guard problem.queries.count == expectedQuery,
          problem.keys.count == expectedKV,
          problem.values.count == expectedKV else {
        throw NSError(domain: "prefix_attention", code: 18,
                      userInfo: [NSLocalizedDescriptionKey: "host buffers do not match case"])
    }
    guard problem.tiles.count == Int(tileCount(spec)) else {
        throw NSError(domain: "prefix_attention", code: 2,
                      userInfo: [NSLocalizedDescriptionKey: "wrong tile count"])
    }
    var next: UInt32 = 0
    for (index, tile) in problem.tiles.enumerated() {
        guard tile.absoluteStart == next, tile.validTokens > 0,
              tile.validTokens <= spec.tileTokens,
              tile.absoluteStart < spec.tokens,
              tile.validTokens <= spec.tokens - tile.absoluteStart,
              UInt64(tile.keyOffsetElements) ==
                  UInt64(tile.absoluteStart) * UInt64(spec.headDim),
              UInt64(tile.valueOffsetElements) ==
                  UInt64(tile.absoluteStart) * UInt64(spec.headDim),
              index + 1 == problem.tiles.count || tile.validTokens == spec.tileTokens else {
            throw NSError(domain: "prefix_attention", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "invalid tile table"])
        }
        next += tile.validTokens
    }
    guard next == spec.tokens else {
        throw NSError(domain: "prefix_attention", code: 4,
                      userInfo: [NSLocalizedDescriptionKey: "tile table does not cover tokens"])
    }
    try validateSchedule(spec, problem.scheduleA, problem.rowIdsA,
                         problem.scheduleTilesA)
    try validateSchedule(spec, problem.scheduleB, problem.rowIdsB,
                         problem.scheduleTilesB)
}

private func validateScheduleRows(_ spec: CaseSpec, _ groups: [QueryGroup],
                                  _ rowIds: [UInt32], _ rowsPerGroup: UInt32,
                                  _ tileCountValue: Int) throws -> Set<UInt32> {
    var seen = Set<UInt32>()
    for group in groups {
        guard group.rowCount > 0, group.rowCount <= rowsPerGroup,
              group.queryHead < spec.queryHeads,
              group.tileCount == tileCount(spec),
              Int(group.rowOffset) + Int(group.rowCount) <= rowIds.count,
              Int(group.tileOffset) + Int(group.tileCount) <= tileCountValue else {
            throw NSError(domain: "prefix_attention", code: 6,
                          userInfo: [NSLocalizedDescriptionKey: "invalid schedule descriptor"])
        }
        let expectedKVHead = group.queryHead * spec.kvHeads / spec.queryHeads
        guard group.kvHead == expectedKVHead else {
            throw NSError(domain: "prefix_attention", code: 19,
                          userInfo: [NSLocalizedDescriptionKey: "invalid schedule head mapping"])
        }
        for offset in 0..<group.rowCount {
            let row = rowIds[Int(group.rowOffset) + Int(offset)]
            guard row < spec.queryRows,
                  seen.insert(row * spec.queryHeads + group.queryHead).inserted else {
                throw NSError(domain: "prefix_attention", code: 7,
                              userInfo: [NSLocalizedDescriptionKey: "repeated schedule row"])
            }
        }
    }
    return seen
}

private func validateScheduledTiles(_ spec: CaseSpec, _ groups: [QueryGroup],
                                    _ tiles: [PrefixTile]) throws {
    for group in groups {
        var starts: [UInt32] = []
        for offset in 0..<group.tileCount {
            let tile = tiles[Int(group.tileOffset) + Int(offset)]
            guard tile.validTokens > 0,
                  tile.validTokens <= spec.tileTokens,
                  tile.absoluteStart < spec.tokens,
                  tile.validTokens <= spec.tokens - tile.absoluteStart,
                  UInt64(tile.keyOffsetElements) ==
                      UInt64(tile.absoluteStart) * UInt64(spec.headDim),
                  UInt64(tile.valueOffsetElements) ==
                      UInt64(tile.absoluteStart) * UInt64(spec.headDim),
                  tile.validTokens == spec.tileTokens ||
                      tile.validTokens == spec.tokens - tile.absoluteStart else {
                throw NSError(domain: "prefix_attention", code: 16,
                              userInfo: [NSLocalizedDescriptionKey: "invalid scheduled tile"])
            }
            starts.append(tile.absoluteStart)
        }
        starts.sort()
        for index in 0..<Int(tileCount(spec)) {
            guard starts[index] == UInt32(index) * spec.tileTokens else {
                throw NSError(domain: "prefix_attention", code: 17,
                              userInfo: [NSLocalizedDescriptionKey: "scheduled tile omitted"])
            }
        }
    }
}

private func roundedScheduleRows(_ spec: CaseSpec) -> UInt32 {
    let rowsPerGroup = effectiveGroupRows(spec)
    return spec.queryRows / rowsPerGroup +
        (spec.queryRows % rowsPerGroup == 0 ? 0 : 1)
}

private func expectedScheduleGroups(_ spec: CaseSpec) -> Int {
    Int(roundedScheduleRows(spec) * spec.queryHeads)
}

private func validateScheduleShape(_ spec: CaseSpec, _ groups: [QueryGroup],
                                   _ rowIds: [UInt32], _ tiles: [PrefixTile]) throws
    -> UInt32 {
    guard groups.count == expectedScheduleGroups(spec) else {
        throw NSError(domain: "prefix_attention", code: 5,
                      userInfo: [NSLocalizedDescriptionKey: "invalid schedule size"])
    }
    guard tiles.count == groups.count * Int(tileCount(spec)) else {
        throw NSError(domain: "prefix_attention", code: 5,
                      userInfo: [NSLocalizedDescriptionKey: "invalid schedule size"])
    }
    let expectedRowIds = groups.reduce(0) { $0 + Int($1.rowCount) }
    guard rowIds.count == expectedRowIds else {
        throw NSError(domain: "prefix_attention", code: 20,
                      userInfo: [NSLocalizedDescriptionKey: "invalid schedule rows"])
    }
    return effectiveGroupRows(spec)
}

private func validateScheduleCoverage(_ spec: CaseSpec, _ seen: Set<UInt32>) throws {
    guard seen.count == Int(spec.queryRows * spec.queryHeads) else {
        throw NSError(domain: "prefix_attention", code: 8,
                      userInfo: [NSLocalizedDescriptionKey: "omitted schedule row"])
    }
}

private func validateSchedule(_ spec: CaseSpec, _ groups: [QueryGroup],
                              _ rowIds: [UInt32], _ tiles: [PrefixTile]) throws {
    let rowsPerGroup = try validateScheduleShape(spec, groups, rowIds, tiles)
    let seen = try validateScheduleRows(spec, groups, rowIds, rowsPerGroup, tiles.count)
    try validateScheduleCoverage(spec, seen)
    try validateScheduledTiles(spec, groups, tiles)
}

private func rejectsMissingTile(_ problem: Problem) -> Bool {
    var malformed = problem
    malformed.tiles.removeLast()
    do {
        try validate(malformed)
        return false
    } catch {
        return true
    }
}

private func rejectsInteriorPartialTile(_ problem: Problem) -> Bool {
    guard problem.tiles.count > 1 else { return false }
    var malformed = problem
    malformed.tiles[0].validTokens -= 1
    do {
        try validate(malformed)
        return false
    } catch {
        return true
    }
}

private func rejectsScheduledTile(_ problem: Problem) -> Bool {
    var malformed = problem
    guard !malformed.scheduleTilesA.isEmpty,
          malformed.scheduleTilesA[0].validTokens > 0 else { return false }
    malformed.scheduleTilesA[0].validTokens -= 1
    do {
        try validate(malformed)
        return false
    } catch {
        return true
    }
}

private func queryKVBase(_ spec: CaseSpec, _ row: UInt32) -> UInt32 {
    spec.sharedPrefix ? 0 : row * spec.kvHeads * spec.tokens * spec.headDim
}

private func oracle(_ problem: Problem) -> [Double] {
    let spec = problem.spec
    let queryStride = spec.queryHeads * spec.headDim
    var result = [Double](repeating: 0, count: Int(spec.queryRows * queryStride))
    for row in 0..<spec.queryRows {
        for queryHead in 0..<spec.queryHeads {
            let kvHead = queryHead * spec.kvHeads / spec.queryHeads
            let queryOffset = Int(row * queryStride + queryHead * spec.headDim)
            let kvBase = queryKVBase(spec, row) + kvHead * spec.tokens * spec.headDim
            let scale = 1.0 / sqrt(Double(spec.headDim))
            var scores = [Double](repeating: 0, count: Int(spec.tokens))
            var maximum = -Double.infinity
            for token in 0..<spec.tokens {
                let offset = Int(kvBase + token * spec.headDim)
                var score = 0.0
                for index in 0..<spec.headDim {
                    score += Double(problem.queries[queryOffset + Int(index)]) *
                        Double(Float(problem.keys[offset + Int(index)]))
                }
                scores[Int(token)] = score * scale
                maximum = max(maximum, scores[Int(token)])
            }
            var normalizer = 0.0
            for token in 0..<spec.tokens {
                let weight = exp(scores[Int(token)] - maximum)
                normalizer += weight
                let valueOffset = Int(kvBase + token * spec.headDim)
                for index in 0..<spec.headDim {
                    result[queryOffset + Int(index)] += weight *
                        Double(Float(problem.values[valueOffset + Int(index)]))
                }
            }
            for index in 0..<spec.headDim {
                result[queryOffset + Int(index)] /= normalizer
            }
        }
    }
    return result
}

private func maxAbsolute(_ output: [Float], _ expected: [Double]) -> Double {
    zip(output, expected).map { abs(Double($0.0) - $0.1) }.max() ?? 0
}

private func maxRelative(_ output: [Float], _ expected: [Double]) -> Double {
    zip(output, expected).map {
        abs(Double($0.0) - $0.1) / max(abs($0.1), 1.0e-12)
    }.max() ?? 0
}

private func digest(_ output: [Float]) -> String {
    var hash: UInt64 = 1469598103934665603
    for value in output {
        var bits = value.bitPattern
        for _ in 0..<4 {
            hash ^= UInt64(bits & 0xff)
            hash &*= 1099511628211
            bits >>= 8
        }
    }
    return String(format: "%016llx", hash)
}

private func updateDigest(_ hash: inout UInt64, _ value: UInt32) {
    var bits = value
    for _ in 0..<4 {
        hash ^= UInt64(bits & 0xff)
        hash &*= 1099511628211
        bits >>= 8
    }
}

private func updateDigest(_ hash: inout UInt64, _ value: UInt16) {
    var bits = value
    for _ in 0..<2 {
        hash ^= UInt64(bits & 0xff)
        hash &*= 1099511628211
        bits >>= 8
    }
}

private func inputDigest(_ problem: Problem) -> String {
    var hash: UInt64 = 1469598103934665603
    for value in problem.queries {
        updateDigest(&hash, value.bitPattern)
    }
    for value in problem.keys {
        updateDigest(&hash, value.bitPattern)
    }
    for value in problem.values {
        updateDigest(&hash, value.bitPattern)
    }
    return String(format: "%016llx", hash)
}

private func oracleDigest(_ values: [Double]) -> String {
    var hash: UInt64 = 1469598103934665603
    for value in values {
        var bits = value.bitPattern
        for _ in 0..<8 {
            hash ^= bits & 0xff
            hash &*= 1099511628211
            bits >>= 8
        }
    }
    return String(format: "%016llx", hash)
}

private func buffer<T>(_ device: MTLDevice, _ values: [T]) -> MTLBuffer {
    values.withUnsafeBufferPointer { pointer in
        device.makeBuffer(bytes: pointer.baseAddress!,
                          length: pointer.count * MemoryLayout<T>.stride,
                          options: .storageModeShared)!
    }
}

private func outputValues(_ buffer: MTLBuffer, _ count: Int) -> [Float] {
    Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self),
                              count: count))
}

private func scheduleData(_ problem: Problem, _ path: Path, _ scheduleB: Bool) ->
    ([PrefixTile], [QueryGroup], [UInt32]) {
    let tiles = path == .sharedReadUnconstrained && scheduleB ? problem.scheduleTilesB :
        path == .sharedReadUnconstrained ? problem.scheduleTilesA : problem.tiles
    var groups = scheduleB ? problem.scheduleB : problem.scheduleA
    if path != .sharedReadUnconstrained {
        for index in groups.indices {
            groups[index].tileOffset = 0
        }
    }
    let rowIds = scheduleB ? problem.rowIdsB : problem.rowIdsA
    return (tiles, groups, rowIds)
}

private func executeCommand(_ queue: MTLCommandQueue,
                            _ pipeline: MTLComputePipelineState,
                            _ path: Path, _ sharedPath: Bool,
                            _ queryBuffer: MTLBuffer, _ keyBuffer: MTLBuffer,
                            _ valueBuffer: MTLBuffer, _ tileBuffer: MTLBuffer,
                            _ groupBuffer: MTLBuffer, _ rowBuffer: MTLBuffer,
                            _ outputBuffer: MTLBuffer, _ parameters: inout Parameters,
                            _ spec: CaseSpec, _ groupsToRun: Int) throws -> Double {
    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw NSError(domain: "prefix_attention", code: 9,
                      userInfo: [NSLocalizedDescriptionKey: "cannot create Metal command"])
    }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(queryBuffer, offset: 0, index: 0)
    encoder.setBuffer(keyBuffer, offset: 0, index: 1)
    encoder.setBuffer(valueBuffer, offset: 0, index: 2)
    encoder.setBuffer(outputBuffer, offset: 0, index: 3)
    encoder.setBuffer(tileBuffer, offset: 0, index: 4)
    encoder.setBuffer(groupBuffer, offset: 0, index: 5)
    encoder.setBuffer(rowBuffer, offset: 0, index: 6)
    if sharedPath {
        encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 7)
        encoder.setThreadgroupMemoryLength(
            2 * Int(spec.tileTokens * spec.headDim) * MemoryLayout<Float16>.stride,
            index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: groupsToRun, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupWidth, height: 1, depth: 1))
    } else {
        let parameterIndex = path == .fixedTilePerRow ? 5 : 4
        encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride,
                         index: parameterIndex)
        let workgroups = (groupsToRun + threadgroupWidth - 1) / threadgroupWidth
        encoder.dispatchThreadgroups(
            MTLSize(width: workgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupWidth, height: 1, depth: 1))
    }
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    guard commandBuffer.status == .completed else {
        throw commandBuffer.error ?? NSError(domain: "prefix_attention", code: 10,
                                              userInfo: [NSLocalizedDescriptionKey: "Metal command failed"])
    }
    return (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1000.0
}

private func runPath(_ device: MTLDevice, _ queue: MTLCommandQueue,
                    _ pipeline: MTLComputePipelineState, _ problem: Problem,
                    _ path: Path, _ scheduleB: Bool, _ warmups: Int,
                    _ repetitions: Int) throws -> RunResult {
    let spec = problem.spec
    let (tiles, groups, rowIds) = scheduleData(problem, path, scheduleB)
    let queryBuffer = buffer(device, problem.queries)
    let keyBuffer = buffer(device, problem.keys)
    let valueBuffer = buffer(device, problem.values)
    let tileBuffer = buffer(device, tiles)
    let groupBuffer = buffer(device, groups)
    let rowBuffer = buffer(device, rowIds)
    let outputCount = Int(spec.queryRows * spec.queryHeads * spec.headDim)
    let outputBuffer = device.makeBuffer(length: outputCount * MemoryLayout<Float>.stride,
                                         options: .storageModeShared)!
    var parameters = Parameters(queryRows: spec.queryRows,
                                queryHeads: spec.queryHeads, kvHeads: spec.kvHeads,
                                tokens: spec.tokens, headDim: spec.headDim,
                                tileCapacityTokens: spec.tileTokens,
                                kvStride: spec.kvHeads * spec.tokens * spec.headDim,
                                sharedPrefix: spec.sharedPrefix ? 1 : 0)
    let sharedPath = path == .sharedReadUnconstrained || path == .sharedReadFixedReduction
    let groupsToRun = sharedPath ? groups.count : Int(spec.queryRows * spec.queryHeads)
    for _ in 0..<warmups {
        _ = try executeCommand(queue, pipeline, path, sharedPath, queryBuffer,
                                keyBuffer, valueBuffer, tileBuffer, groupBuffer,
                                rowBuffer, outputBuffer, &parameters, spec, groupsToRun)
    }
    var samples: [Double] = []
    for _ in 0..<repetitions {
        samples.append(try executeCommand(queue, pipeline, path, sharedPath,
                                          queryBuffer, keyBuffer, valueBuffer,
                                          tileBuffer, groupBuffer, rowBuffer,
                                          outputBuffer, &parameters, spec,
                                          groupsToRun))
    }
    let sortedSamples = samples.sorted()
    let estimatedGroups = sharedPath ? groups.count : Int(spec.queryRows * spec.queryHeads)
    return RunResult(
        output: outputValues(outputBuffer, outputCount), samplesMS: samples,
        medianMS: sortedSamples[sortedSamples.count / 2],
        perBlockSharedBytes: sharedPath ? UInt64(2 * spec.tileTokens * spec.headDim * 2) : 0,
        estimatedKVElementsRead: UInt64(estimatedGroups) * UInt64(spec.tokens) *
            UInt64(spec.headDim) * 2,
        estimatedTileLoads: UInt64(estimatedGroups) * UInt64(tileCount(spec)),
        outputElementsWritten: UInt64(outputCount))
}

private func bitwiseEqual(_ left: [Float], _ right: [Float]) -> Bool {
    left.count == right.count && zip(left, right).allSatisfy { $0.0.bitPattern == $0.1.bitPattern }
}

private func timingPair(_ baselineBefore: RunResult, _ candidate: RunResult,
                        _ baselineAfter: RunResult) -> [String: Any] {
    [
        "baseline_path": "fixed_tile_per_row",
        "candidate_path": "shared_read_fixed_reduction",
        "acquisition_order": ["baseline_before", "candidate", "baseline_after"],
        "baseline_before_samples_ms": baselineBefore.samplesMS,
        "candidate_samples_ms": candidate.samplesMS,
        "baseline_after_samples_ms": baselineAfter.samplesMS,
        "baseline_before_median_ms": baselineBefore.medianMS,
        "candidate_median_ms": candidate.medianMS,
        "baseline_after_median_ms": baselineAfter.medianMS,
    ]
}

private struct Options {
    var phase = ""
    var metallib = ""
    var output = ""
    var warmups = 3
    var repetitions = 9
}

private func parseOption(_ argument: String, _ value: String,
                         _ options: inout Options) throws {
    switch argument {
    case "--phase": options.phase = value
    case "--metallib": options.metallib = value
    case "--output": options.output = value
    case "--warmups": options.warmups = Int(value) ?? -1
    case "--repetitions": options.repetitions = Int(value) ?? -1
    default:
        throw NSError(domain: "prefix_attention", code: 12,
                      userInfo: [NSLocalizedDescriptionKey: "unknown option \(argument)"])
    }
}

private func validateOptions(_ options: Options) throws {
    guard options.phase == "calibration" || options.phase == "evaluation",
          !options.metallib.isEmpty, !options.output.isEmpty,
          options.warmups >= 0, options.repetitions > 0 else {
        throw NSError(domain: "prefix_attention", code: 13,
                      userInfo: [NSLocalizedDescriptionKey: "invalid options"])
    }
}

private func parseOptions() throws -> Options {
    var options = Options()
    var index = 1
    let arguments = CommandLine.arguments
    while index < arguments.count {
        guard index + 1 < arguments.count else {
            throw NSError(domain: "prefix_attention", code: 11,
                          userInfo: [NSLocalizedDescriptionKey: "option needs a value"])
        }
        try parseOption(arguments[index], arguments[index + 1], &options)
        index += 2
    }
    try validateOptions(options)
    return options
}

private func functionName(_ path: Path) -> String {
    switch path {
    case .perRow: return "per_row_kernel"
    case .fixedTilePerRow: return "fixed_tile_per_row_kernel"
    case .sharedReadUnconstrained: return "shared_read_unconstrained_kernel"
    case .sharedReadFixedReduction: return "shared_read_fixed_reduction_kernel"
    }
}

private func makePipelines(_ device: MTLDevice, _ library: MTLLibrary) throws
    -> [Path: MTLComputePipelineState] {
    var pipelines: [Path: MTLComputePipelineState] = [:]
    for path in Path.allCases {
        let name = functionName(path)
        guard let function = library.makeFunction(name: name) else {
            throw NSError(domain: "prefix_attention", code: 15,
                          userInfo: [NSLocalizedDescriptionKey: "missing Metal function \(name)"])
        }
        pipelines[path] = try device.makeComputePipelineState(function: function)
    }
    return pipelines
}

private func runCase(_ device: MTLDevice, _ queue: MTLCommandQueue,
                     _ pipelines: [Path: MTLComputePipelineState],
                     _ spec: CaseSpec, _ options: Options) throws -> [String: Any] {
    let problem = try makeProblem(spec)
    let oracleValues = oracle(problem)
    var pathReceipts: [[String: Any]] = []
    for path in Path.allCases {
        let runResult = try runPath(device, queue, pipelines[path]!, problem, path,
                                    false, options.warmups, options.repetitions)
        let scheduleB = try runPath(device, queue, pipelines[path]!, problem, path,
                                    true, 0, 1)
        pathReceipts.append([
            "path": path.rawValue,
            "samples_ms": runResult.samplesMS,
            "median_ms": runResult.medianMS,
            "quality_max_abs": maxAbsolute(runResult.output, oracleValues),
            "quality_max_rel": maxRelative(runResult.output, oracleValues),
            "output_values": runResult.output.map(Double.init),
            "schedule_b_output_values": scheduleB.output.map(Double.init),
            "schedule_b_quality_max_abs": maxAbsolute(scheduleB.output, oracleValues),
            "schedule_b_quality_max_rel": maxRelative(scheduleB.output, oracleValues),
            "digest": digest(runResult.output),
            "schedule_b_digest": digest(scheduleB.output),
            "schedule_b_bitwise_equal": bitwiseEqual(runResult.output, scheduleB.output),
            "estimated_kv_elements_read": runResult.estimatedKVElementsRead,
            "estimated_tile_loads": runResult.estimatedTileLoads,
            "output_elements_written": runResult.outputElementsWritten,
            "per_block_shared_bytes": runResult.perBlockSharedBytes,
        ])
    }
    let baselineBefore = try runPath(
        device, queue, pipelines[.fixedTilePerRow]!, problem,
        .fixedTilePerRow, false, options.warmups, options.repetitions)
    let candidate = try runPath(
        device, queue, pipelines[.sharedReadFixedReduction]!, problem,
        .sharedReadFixedReduction, false, options.warmups, options.repetitions)
    let baselineAfter = try runPath(
        device, queue, pipelines[.fixedTilePerRow]!, problem,
        .fixedTilePerRow, false, options.warmups, options.repetitions)
    return [
        "name": spec.name,
        "input_digest": inputDigest(problem),
        "oracle_digest": oracleDigest(oracleValues),
        "spec": ["tokens": spec.tokens, "query_rows": spec.queryRows,
                 "query_heads": spec.queryHeads, "kv_heads": spec.kvHeads,
                 "head_dim": spec.headDim, "tile_tokens": spec.tileTokens,
                 "group_rows": spec.groupRows, "seed": spec.seed,
                 "shared_prefix": spec.sharedPrefix],
        "partial_final_tile": problem.tiles.last!.validTokens < spec.tileTokens,
        "missing_tile_rejected": rejectsMissingTile(problem),
        "interior_partial_tile_rejected": rejectsInteriorPartialTile(problem),
        "scheduled_tile_rejected": rejectsScheduledTile(problem),
        "paths": pathReceipts,
        "timing_pair": timingPair(baselineBefore, candidate, baselineAfter),
    ]
}

private func run(_ options: Options) throws {
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue(),
          let library = try? device.makeLibrary(filepath: options.metallib) else {
        throw NSError(domain: "prefix_attention", code: 14,
                      userInfo: [NSLocalizedDescriptionKey: "cannot create Metal device or library"])
    }
    let pipelines = try makePipelines(device, library)
    let cases = options.phase == "calibration" ? manifestCalibrationCases() : manifestEvaluationCases()
    var receipt: [String: Any] = [
        "schema": "prefix-attention-gpu-receipt-v1",
        "operator": "fixed-reduction-shared-prefix-attention",
        "storage": "fp16-kv-fp32-accumulation",
        "backend": "metal",
        "phase": options.phase,
        "input_generator": manifestInputGenerator,
        "input_digest_algorithm": manifestInputDigestAlgorithm,
        "oracle_id": manifestOracleID,
        "unavailable_evidence": ["host_launch_time", "dram_traffic", "power", "clocks"],
        "claims": [],
        "device": ["name": device.name, "registry_id": device.registryID],
        "warmups": options.warmups,
        "repetitions": options.repetitions,
    ]
    var caseReceipts: [[String: Any]] = []
    for spec in cases {
        caseReceipts.append(try runCase(device, queue, pipelines, spec, options))
    }
    receipt["cases"] = caseReceipts
    let data = try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys])
    try FileManager.default.createDirectory(
        at: URL(fileURLWithPath: options.output).deletingLastPathComponent(),
        withIntermediateDirectories: true, attributes: nil)
    try data.write(to: URL(fileURLWithPath: options.output))
}

do {
    try run(parseOptions())
} catch {
    FileHandle.standardError.write(Data("prefix_attention_metal: \(error)\n".utf8))
    exit(1)
}
