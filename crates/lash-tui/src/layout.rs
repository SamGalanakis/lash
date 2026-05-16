use crate::Rect;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    Horizontal,
    Vertical,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Constraint {
    Length(u16),
    Min(u16),
    Fill(u16),
}

pub struct Layout;

impl Layout {
    pub fn split(area: Rect, axis: Axis, constraints: &[Constraint]) -> Vec<Rect> {
        if constraints.is_empty() {
            return Vec::new();
        }

        let total = match axis {
            Axis::Horizontal => area.width,
            Axis::Vertical => area.height,
        } as usize;

        let mut sizes = vec![0usize; constraints.len()];
        let mut remaining = total;

        for (index, constraint) in constraints.iter().enumerate() {
            if let Constraint::Length(length) = *constraint {
                let size = (length as usize).min(remaining);
                sizes[index] = size;
                remaining = remaining.saturating_sub(size);
            }
        }

        for (index, constraint) in constraints.iter().enumerate() {
            if let Constraint::Min(minimum) = *constraint {
                let size = (minimum as usize).min(remaining);
                sizes[index] = size;
                remaining = remaining.saturating_sub(size);
            }
        }

        let fill_indices = constraints
            .iter()
            .enumerate()
            .filter_map(|(index, constraint)| match *constraint {
                Constraint::Fill(weight) if weight > 0 => Some((index, weight as usize)),
                _ => None,
            })
            .collect::<Vec<_>>();

        if remaining > 0 && !fill_indices.is_empty() {
            let total_weight = fill_indices
                .iter()
                .map(|(_, weight)| *weight)
                .sum::<usize>()
                .max(1);
            let mut distributed = 0usize;
            for (index, weight) in &fill_indices {
                let share = remaining * *weight / total_weight;
                sizes[*index] = sizes[*index].saturating_add(share);
                distributed = distributed.saturating_add(share);
            }

            let mut remainder = remaining.saturating_sub(distributed);
            for (index, _) in &fill_indices {
                if remainder == 0 {
                    break;
                }
                sizes[*index] = sizes[*index].saturating_add(1);
                remainder -= 1;
            }
        }

        let mut offset = 0u16;
        sizes
            .into_iter()
            .map(|size| {
                let size = size as u16;
                let rect = match axis {
                    Axis::Horizontal => Rect::new(
                        area.x.saturating_add(offset),
                        area.y,
                        size.min(area.width.saturating_sub(offset)),
                        area.height,
                    ),
                    Axis::Vertical => Rect::new(
                        area.x,
                        area.y.saturating_add(offset),
                        area.width,
                        size.min(area.height.saturating_sub(offset)),
                    ),
                };
                offset = offset.saturating_add(size);
                rect
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{Axis, Constraint, Layout};
    use crate::Rect;

    #[test]
    fn vertical_layout_respects_fixed_and_fill() {
        let parts = Layout::split(
            Rect::new(0, 0, 20, 10),
            Axis::Vertical,
            &[
                Constraint::Length(2),
                Constraint::Fill(1),
                Constraint::Length(1),
            ],
        );
        assert_eq!(parts[0], Rect::new(0, 0, 20, 2));
        assert_eq!(parts[1], Rect::new(0, 2, 20, 7));
        assert_eq!(parts[2], Rect::new(0, 9, 20, 1));
    }

    #[test]
    fn horizontal_layout_respects_minimums() {
        let parts = Layout::split(
            Rect::new(0, 0, 9, 4),
            Axis::Horizontal,
            &[Constraint::Min(3), Constraint::Min(4), Constraint::Fill(1)],
        );
        assert_eq!(parts[0].width, 3);
        assert_eq!(parts[1].width, 4);
        assert_eq!(parts[2].width, 2);
    }

    #[test]
    fn layout_never_exceeds_parent_rect() {
        let area = Rect::new(0, 0, 5, 5);
        let parts = Layout::split(
            area,
            Axis::Horizontal,
            &[Constraint::Length(10), Constraint::Fill(1)],
        );
        assert_eq!(parts[0].width + parts[1].width, area.width);
    }

    #[test]
    fn layout_distributes_remainder_stably() {
        let parts = Layout::split(
            Rect::new(0, 0, 5, 1),
            Axis::Horizontal,
            &[Constraint::Fill(1), Constraint::Fill(1)],
        );
        assert_eq!(parts[0].width, 3);
        assert_eq!(parts[1].width, 2);
    }
}
