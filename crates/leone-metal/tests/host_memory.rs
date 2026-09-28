#![cfg(target_os = "macos")]

use std::process::Command;

#[test]
fn host_memory_snapshot_matches_macos_memory_contract() {
    let snapshot = leone_metal::host_memory_snapshot().expect("macOS host memory bridge");
    let independent_total = independent_memsize_bytes();
    let total = snapshot.system_total_bytes.expect("host memory total");
    assert_eq!(total, independent_total);

    let resident = snapshot
        .process_resident_bytes
        .expect("process resident bytes");
    let virtual_bytes = snapshot
        .process_virtual_bytes
        .expect("process virtual bytes");
    assert!(resident > 0);
    assert!(virtual_bytes >= resident);

    let available = snapshot
        .system_available_bytes
        .expect("system available bytes");
    let used = snapshot.system_used_bytes.expect("system used bytes");
    assert!(available <= total);
    assert!(used <= total);
    assert_eq!(used.checked_add(available), Some(total));
}

fn independent_memsize_bytes() -> u64 {
    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .expect("sysctl hw.memsize");
    assert!(output.status.success());
    String::from_utf8(output.stdout)
        .expect("sysctl output")
        .trim()
        .parse()
        .expect("sysctl memory size")
}
