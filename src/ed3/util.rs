pub(super) fn duplicate_elements<'a, I, T>(iter: I) -> impl Iterator<Item = T> + 'a
where
    I: Iterator<Item = &'a T> + 'a,
    T: Copy + 'a,
{
    iter.flat_map(|&item| std::iter::repeat(item).take(2))
}
