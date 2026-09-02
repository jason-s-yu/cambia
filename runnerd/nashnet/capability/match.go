package capability

// Reason names one constraint Match found unsatisfied. D14 renders the union
// of a job's match rejection reasons across every candidate node in its
// JobView and the queue snapshot.
type Reason string

const (
	// ReasonKindUnsupported means the job's kind is not in the candidate's
	// (clamped) declared kinds list.
	ReasonKindUnsupported Reason = "kind_unsupported"
	// ReasonDeviceUnavailable means the candidate declares no device of the
	// required kind at all.
	ReasonDeviceUnavailable Reason = "device_unavailable"
	// ReasonCapabilityUndeclared means a requires floor reads a fact the node
	// never declared (D9): the field is absent, not a fabricated zero.
	ReasonCapabilityUndeclared Reason = "capability_undeclared"
	ReasonMinVRAMGB            Reason = "min_vram_gb"
	ReasonMinCores             Reason = "min_cores"
	ReasonMinRAMGB             Reason = "min_ram_gb"
	ReasonMinDiskGB            Reason = "min_disk_gb"
	ReasonNeedsLibcambia       Reason = "needs_libcambia"
	ReasonNodeMismatch         Reason = "node"
	ReasonLabelsAny            Reason = "labels_any"
)

// Candidate is one node's identity and its already grant-clamped declaration
// (Clamp must run before Match sees it), plus the grant's own label set:
// labels_any (D10) matches against the grant's Labels, never the node's
// self-declared Declaration.Labels, which Clamp already drops (D47).
type Candidate struct {
	NodeID      string
	Declaration Declaration
	GrantLabels []string
}

// Match reports whether a job of the given kind, with an already-normalized
// requires block (call Requires.Normalize first), may place on candidate. It
// is a pure function over declared and reported facts only -- no clock, no
// I/O -- reading only the clamped declaration and the grant's label set
// (D11). false is always paired with at least one Reason; the coordinator
// never substitutes an assumed value for a field the node did not declare.
func Match(kind string, requires Requires, candidate Candidate) (bool, []Reason) {
	var reasons []Reason
	d := candidate.Declaration

	if !containsString(d.Kinds, kind) {
		reasons = append(reasons, ReasonKindUnsupported)
	}

	matches := devicesOfKind(d.Devices, requires.Device)
	if len(matches) == 0 {
		reasons = append(reasons, ReasonDeviceUnavailable)
	} else if requires.MinVRAMGB != nil {
		declared, satisfied := false, false
		for _, dev := range matches {
			if dev.VRAMTotalGB == nil {
				continue
			}
			declared = true
			if *dev.VRAMTotalGB >= *requires.MinVRAMGB {
				satisfied = true
				break
			}
		}
		switch {
		case !declared:
			reasons = append(reasons, ReasonCapabilityUndeclared)
		case !satisfied:
			reasons = append(reasons, ReasonMinVRAMGB)
		}
	}

	if requires.MinCores > 0 {
		cpu, hasCPU := deviceOfKind(d.Devices, "cpu")
		switch {
		case !hasCPU || cpu.Cores == nil:
			reasons = append(reasons, ReasonCapabilityUndeclared)
		case *cpu.Cores < requires.MinCores:
			reasons = append(reasons, ReasonMinCores)
		}
	}

	if requires.MinRAMGB != nil {
		cpu, hasCPU := deviceOfKind(d.Devices, "cpu")
		switch {
		case !hasCPU || cpu.RAMTotalGB == nil:
			reasons = append(reasons, ReasonCapabilityUndeclared)
		case *cpu.RAMTotalGB < *requires.MinRAMGB:
			reasons = append(reasons, ReasonMinRAMGB)
		}
	}

	if requires.MinDiskGB != nil {
		switch {
		case d.DiskTotalGB == nil:
			reasons = append(reasons, ReasonCapabilityUndeclared)
		case *d.DiskTotalGB < *requires.MinDiskGB:
			reasons = append(reasons, ReasonMinDiskGB)
		}
	}

	if requires.NeedsLibcambia != nil && *requires.NeedsLibcambia && !d.CanBuildLibcambia {
		reasons = append(reasons, ReasonNeedsLibcambia)
	}

	if requires.Node != "" && requires.Node != candidate.NodeID {
		reasons = append(reasons, ReasonNodeMismatch)
	}

	if len(requires.LabelsAny) > 0 && !intersects(requires.LabelsAny, candidate.GrantLabels) {
		reasons = append(reasons, ReasonLabelsAny)
	}

	return len(reasons) == 0, reasons
}
