use std::collections::BTreeMap;

use lash_tui::{InputEvent, Rect};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UiSurfaceSlot {
    #[default]
    Workspace,
    Dock,
    Footer,
    Overlay,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UiSurfaceSize {
    #[default]
    Auto,
    Lines(u16),
    Fixed {
        width: u16,
        height: u16,
    },
}

impl UiSurfaceSize {
    pub const fn height(self) -> u16 {
        match self {
            Self::Auto => 1,
            Self::Lines(lines) => lines,
            Self::Fixed { height, .. } => height,
        }
    }

    pub const fn width(self) -> Option<u16> {
        match self {
            Self::Fixed { width, .. } => Some(width),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UiSurfaceSpec {
    pub key: String,
    pub slot: UiSurfaceSlot,
    pub size: UiSurfaceSize,
    pub order: i32,
    pub focusable: bool,
    pub visible: bool,
    pub modal: bool,
}

impl UiSurfaceSpec {
    pub fn new(key: impl Into<String>, slot: UiSurfaceSlot) -> Self {
        Self {
            key: key.into(),
            slot,
            size: UiSurfaceSize::Auto,
            order: 0,
            focusable: false,
            visible: true,
            modal: false,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UiSurfaceUpdate {
    pub slot: Option<UiSurfaceSlot>,
    pub size: Option<UiSurfaceSize>,
    pub order: Option<i32>,
    pub focusable: Option<bool>,
    pub visible: Option<bool>,
    pub modal: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UiMountedSurface {
    pub id: String,
    pub owner_id: String,
    pub key: String,
    pub slot: UiSurfaceSlot,
    pub size: UiSurfaceSize,
    pub order: i32,
    pub focusable: bool,
    pub visible: bool,
    pub modal: bool,
    pub area: Option<Rect>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UiSurfaceScene {
    pub focused: Option<String>,
    pub workspace: Vec<UiMountedSurface>,
    pub dock: Vec<UiMountedSurface>,
    pub footer: Vec<UiMountedSurface>,
    pub overlay: Vec<UiMountedSurface>,
}

impl UiSurfaceScene {
    pub fn has_slot(&self, slot: UiSurfaceSlot) -> bool {
        !self.surfaces(slot).is_empty()
    }

    pub fn surfaces(&self, slot: UiSurfaceSlot) -> &[UiMountedSurface] {
        match slot {
            UiSurfaceSlot::Workspace => &self.workspace,
            UiSurfaceSlot::Dock => &self.dock,
            UiSurfaceSlot::Footer => &self.footer,
            UiSurfaceSlot::Overlay => &self.overlay,
        }
    }

    pub fn stack_height(&self, slot: UiSurfaceSlot, max_height: u16) -> u16 {
        if max_height == 0 {
            return 0;
        }
        let mut remaining = max_height;
        let mut total = 0u16;
        for surface in self.surfaces(slot) {
            let height = surface.size.height().min(remaining);
            total = total.saturating_add(height);
            remaining = remaining.saturating_sub(height);
            if remaining == 0 {
                break;
            }
        }
        total
    }
}

#[derive(Clone, Debug)]
struct SurfaceRecord {
    owner_id: String,
    spec: UiSurfaceSpec,
    area: Option<Rect>,
}

impl SurfaceRecord {
    fn mounted(&self) -> UiMountedSurface {
        UiMountedSurface {
            id: global_surface_id(&self.owner_id, &self.spec.key),
            owner_id: self.owner_id.clone(),
            key: self.spec.key.clone(),
            slot: self.spec.slot,
            size: self.spec.size,
            order: self.spec.order,
            focusable: self.spec.focusable,
            visible: self.spec.visible,
            modal: self.spec.modal,
            area: self.area,
        }
    }
}

#[derive(Clone, Default)]
pub struct SurfaceRegistry {
    surfaces: BTreeMap<String, SurfaceRecord>,
    focused: Option<String>,
    focus_stack: Vec<String>,
}

impl SurfaceRegistry {
    pub fn mount(&mut self, owner_id: &str, spec: UiSurfaceSpec) {
        let id = global_surface_id(owner_id, &spec.key);
        self.surfaces.insert(
            id,
            SurfaceRecord {
                owner_id: owner_id.to_string(),
                spec,
                area: None,
            },
        );
        self.prune_focus();
    }

    pub fn update(&mut self, owner_id: &str, key: &str, update: UiSurfaceUpdate) {
        let id = global_surface_id(owner_id, key);
        let Some(record) = self.surfaces.get_mut(&id) else {
            return;
        };
        if let Some(slot) = update.slot {
            record.spec.slot = slot;
        }
        if let Some(size) = update.size {
            record.spec.size = size;
        }
        if let Some(order) = update.order {
            record.spec.order = order;
        }
        if let Some(focusable) = update.focusable {
            record.spec.focusable = focusable;
        }
        if let Some(visible) = update.visible {
            record.spec.visible = visible;
        }
        if let Some(modal) = update.modal {
            record.spec.modal = modal;
        }
        self.prune_focus();
    }

    pub fn unmount(&mut self, owner_id: &str, key: &str) {
        let id = global_surface_id(owner_id, key);
        self.surfaces.remove(&id);
        self.remove_focus_entry(&id);
        self.prune_focus();
    }

    pub fn focus(&mut self, owner_id: &str, key: &str) {
        let id = global_surface_id(owner_id, key);
        let Some(record) = self.surfaces.get(&id) else {
            return;
        };
        if !(record.spec.visible && record.spec.focusable) {
            return;
        }
        if self.focused.as_deref() == Some(id.as_str()) {
            return;
        }
        if let Some(previous) = self.focused.take()
            && previous != id
        {
            self.focus_stack.push(previous);
        }
        self.focused = Some(id);
        self.prune_focus();
    }

    pub fn blur(&mut self, owner_id: &str, key: &str) {
        let id = global_surface_id(owner_id, key);
        self.remove_focus_entry(&id);
        self.prune_focus();
    }

    pub fn focused_surface(&self) -> Option<String> {
        self.focused.clone()
    }

    pub fn has_surface_in_slot(&self, slot: UiSurfaceSlot) -> bool {
        self.surfaces
            .values()
            .any(|record| record.spec.visible && record.spec.slot == slot)
    }

    pub fn surfaces_in_slot(&self, slot: UiSurfaceSlot) -> Vec<UiMountedSurface> {
        self.scene().surfaces(slot).to_vec()
    }

    pub fn stack_height(&self, slot: UiSurfaceSlot, max_height: u16) -> u16 {
        self.scene().stack_height(slot, max_height)
    }

    pub fn clear_areas(&mut self) {
        for record in self.surfaces.values_mut() {
            record.area = None;
        }
    }

    pub fn set_area(&mut self, id: &str, area: Option<Rect>) {
        if let Some(record) = self.surfaces.get_mut(id) {
            record.area = area;
        }
    }

    pub fn surface(&self, id: &str) -> Option<UiMountedSurface> {
        self.surfaces.get(id).map(SurfaceRecord::mounted)
    }

    pub fn scene(&self) -> UiSurfaceScene {
        let mut scene = UiSurfaceScene {
            focused: self.focused.clone(),
            ..UiSurfaceScene::default()
        };
        for record in self.surfaces.values() {
            if !record.spec.visible {
                continue;
            }
            let mounted = record.mounted();
            match mounted.slot {
                UiSurfaceSlot::Workspace => scene.workspace.push(mounted),
                UiSurfaceSlot::Dock => scene.dock.push(mounted),
                UiSurfaceSlot::Footer => scene.footer.push(mounted),
                UiSurfaceSlot::Overlay => scene.overlay.push(mounted),
            }
        }
        for slot_surfaces in [
            &mut scene.workspace,
            &mut scene.dock,
            &mut scene.footer,
            &mut scene.overlay,
        ] {
            slot_surfaces.sort_by(|left, right| {
                left.order
                    .cmp(&right.order)
                    .then_with(|| left.id.cmp(&right.id))
            });
        }
        scene
    }

    pub fn target_for_input(&mut self, event: &InputEvent) -> Option<UiMountedSurface> {
        let target = match event {
            InputEvent::Mouse(mouse) => self.target_for_pointer(mouse.column, mouse.row),
            _ => self.focused_target(),
        }?;
        if matches!(event, InputEvent::Mouse(_)) && target.focusable {
            self.focus(&target.owner_id, &target.key);
        }
        self.surface(&target.id)
    }

    fn focused_target(&mut self) -> Option<UiMountedSurface> {
        self.prune_focus();
        let focused = self.focused.as_ref()?;
        self.surface(focused)
    }

    fn target_for_pointer(&self, column: u16, row: u16) -> Option<UiMountedSurface> {
        let mut best: Option<UiMountedSurface> = None;
        for record in self.surfaces.values() {
            let Some(area) = record.area else {
                continue;
            };
            if !record.spec.visible
                || area.width == 0
                || area.height == 0
                || column < area.x
                || column >= area.right()
                || row < area.y
                || row >= area.bottom()
            {
                continue;
            }
            let candidate = record.mounted();
            let replace = best
                .as_ref()
                .is_none_or(|current| surface_rank(&candidate) > surface_rank(current));
            if replace {
                best = Some(candidate);
            }
        }
        best
    }

    fn remove_focus_entry(&mut self, id: &str) {
        if self.focused.as_deref() == Some(id) {
            self.focused = None;
        }
        self.focus_stack.retain(|entry| entry != id);
    }

    fn prune_focus(&mut self) {
        if self
            .focused
            .as_ref()
            .is_some_and(|id| self.surface_is_focusable(id))
        {
            return;
        }
        self.focused = None;
        while let Some(candidate) = self.focus_stack.pop() {
            if self.surface_is_focusable(&candidate) {
                self.focused = Some(candidate);
                break;
            }
        }
    }

    fn surface_is_focusable(&self, id: &str) -> bool {
        self.surfaces
            .get(id)
            .is_some_and(|record| record.spec.visible && record.spec.focusable)
    }
}

fn slot_priority(slot: UiSurfaceSlot) -> u8 {
    match slot {
        UiSurfaceSlot::Workspace => 0,
        UiSurfaceSlot::Dock => 1,
        UiSurfaceSlot::Footer => 2,
        UiSurfaceSlot::Overlay => 3,
    }
}

fn surface_rank(surface: &UiMountedSurface) -> (u8, i32, &str) {
    (
        slot_priority(surface.slot),
        surface.order,
        surface.id.as_str(),
    )
}

pub fn global_surface_id(owner_id: &str, key: &str) -> String {
    format!("{owner_id}:{key}")
}
