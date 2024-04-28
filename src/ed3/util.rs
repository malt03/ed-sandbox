pub fn duplicate_elements<'a, I, T>(iter: I) -> impl Iterator<Item = T> + 'a
where
    I: Iterator<Item = &'a T> + 'a,
    T: Copy + 'a,
{
    iter.flat_map(|&item| std::iter::repeat(item).take(2))
}

pub fn unduplicate_elements<'a, I, T>(iter: I) -> impl Iterator<Item = T> + 'a
where
    I: Iterator<Item = &'a T> + 'a,
    T: Copy + 'a,
{
    iter.enumerate()
        .filter(|(i, _)| i % 2 == 0)
        .map(|(_, &n)| n)
}
