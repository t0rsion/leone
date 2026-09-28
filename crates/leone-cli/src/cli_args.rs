use std::io;

pub(crate) fn flag_value<'a>(
    arguments: &'a [String],
    index: &mut usize,
) -> Result<&'a str, io::Error> {
    *index += 1;
    arguments.get(*index).map(String::as_str).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "command flag is missing its value",
        )
    })
}
