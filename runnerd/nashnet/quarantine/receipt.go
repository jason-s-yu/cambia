package quarantine

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"time"
)

// ReceiptLine is one line of receipt.jsonl: what the node offered against what
// was promoted or rejected, with a reason for every rejection. It is the audit
// record for a lease and is retained for the debug TTL after the tree is
// otherwise reaped (D59).
type ReceiptLine struct {
	At             int64       `json:"at"`
	Seq            int64       `json:"seq"`
	Digest         string      `json:"digest"`
	Offered        int         `json:"offered"`
	OfferedDeletes int         `json:"offered_deletes"`
	Final          bool        `json:"final"`
	Promoted       []string    `json:"promoted"`
	Deleted        []string    `json:"deleted"`
	Rejected       []Rejection `json:"rejected"`
	Unrecognized   []string    `json:"unrecognized,omitempty"`
	Degraded       bool        `json:"degraded,omitempty"`
}

func receiptLine(at time.Time, req CommitRequest, resp CommitResponse) ReceiptLine {
	return ReceiptLine{
		At:             at.UnixNano(),
		Seq:            resp.Seq,
		Digest:         resp.Digest,
		Offered:        len(req.Entries),
		OfferedDeletes: len(req.Deletes),
		Final:          req.Final,
		Promoted:       resp.Promoted,
		Deleted:        resp.Deleted,
		Rejected:       resp.Rejected,
		Unrecognized:   resp.Unrecognized,
		Degraded:       resp.Degraded,
	}
}

// appendReceipt appends one line and fsyncs it, so the audit record survives a
// crash immediately after a promotion.
func (s *Store) appendReceipt(leaseDir string, line ReceiptLine) error {
	data, err := json.Marshal(line)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	f, err := os.OpenFile(filepath.Join(leaseDir, receiptFile), os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// Receipt reads a lease's receipt lines in order, for the lease view and for
// post-mortem reading of a promotion that rejected something.
func (s *Store) Receipt(l Lease) ([]ReceiptLine, error) {
	dir, err := s.leaseDir(l)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(filepath.Join(dir, receiptFile))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	var out []ReceiptLine
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 8<<20)
	for sc.Scan() {
		if len(sc.Bytes()) == 0 {
			continue
		}
		var line ReceiptLine
		if err := json.Unmarshal(sc.Bytes(), &line); err != nil {
			return out, err
		}
		out = append(out, line)
	}
	return out, sc.Err()
}
