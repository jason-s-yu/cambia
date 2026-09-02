package capability

// Grant is the enrollment grant's caps claim (D60), the ceiling a node's
// self-declared Declaration is clamped to before placement ever reads it
// (D47). An unset field means the declared value passes unclamped -- the
// enrolling operator's explicit choice, not a zero floor.
type Grant struct {
	MaxSlots      *int               `json:"max_slots,omitempty"`
	Kinds         []string           `json:"kinds,omitempty"`
	DeviceKinds   []string           `json:"device_kinds,omitempty"`
	MaxVRAMGB     map[string]float64 `json:"max_vram_gb,omitempty"` // keyed by device kind (cuda, xpu)
	MaxCores      *int               `json:"max_cores,omitempty"`
	MaxRAMGB      *float64           `json:"max_ram_gb,omitempty"`
	MaxDiskGB     *float64           `json:"max_disk_gb,omitempty"`
	Labels        []string           `json:"labels,omitempty"`
	MaxLeaseBytes *int64             `json:"max_lease_bytes,omitempty"`
}

// Clamp applies every row of the D47 table to d and returns the clamped
// declaration placement reads. It never raises a value past what was
// declared -- a grant only lowers a ceiling, it cannot grant a node hardware
// it did not itself report. Labels is dropped from the result entirely: D47
// requires labels_any to match against the grant's own Labels, never the
// node's self-declared set, so nothing here should carry the declared labels
// forward for a caller to accidentally read.
func Clamp(d Declaration, g Grant) Declaration {
	out := d

	if g.MaxSlots != nil && out.Slots > *g.MaxSlots {
		out.Slots = *g.MaxSlots
	}
	if g.Kinds != nil {
		out.Kinds = intersectStrings(out.Kinds, g.Kinds)
	}

	devices := out.Devices
	if g.DeviceKinds != nil {
		devices = filterDeviceKinds(devices, g.DeviceKinds)
	}
	if devices != nil {
		clamped := make([]Device, len(devices))
		for i, dev := range devices {
			clamped[i] = clampDevice(dev, g)
		}
		devices = clamped
	}
	out.Devices = devices

	if g.MaxDiskGB != nil && out.DiskTotalGB != nil && *out.DiskTotalGB > *g.MaxDiskGB {
		v := *g.MaxDiskGB
		out.DiskTotalGB = &v
	}

	out.Labels = nil
	return out
}

// clampDevice applies the vram_total_gb, cores, and ram_total_gb rows of the
// D47 table to one device.
func clampDevice(dev Device, g Grant) Device {
	out := dev
	if g.MaxVRAMGB != nil && out.VRAMTotalGB != nil {
		if capGB, ok := g.MaxVRAMGB[out.Kind]; ok && *out.VRAMTotalGB > capGB {
			v := capGB
			out.VRAMTotalGB = &v
		}
	}
	if g.MaxCores != nil && out.Cores != nil && *out.Cores > *g.MaxCores {
		v := *g.MaxCores
		out.Cores = &v
	}
	if g.MaxRAMGB != nil && out.RAMTotalGB != nil && *out.RAMTotalGB > *g.MaxRAMGB {
		v := *g.MaxRAMGB
		out.RAMTotalGB = &v
	}
	return out
}

func filterDeviceKinds(devices []Device, allowed []string) []Device {
	out := make([]Device, 0, len(devices))
	for _, d := range devices {
		if containsString(allowed, d.Kind) {
			out = append(out, d)
		}
	}
	return out
}

func intersectStrings(declared, allowed []string) []string {
	out := make([]string, 0, len(declared))
	for _, v := range declared {
		if containsString(allowed, v) {
			out = append(out, v)
		}
	}
	return out
}
