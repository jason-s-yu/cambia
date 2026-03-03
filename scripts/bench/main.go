// Lightweight round-trip benchmark for the Cambia stack.
// Measures: HTTP auth, REST API, WS connect, WS message RTT.
//
// Usage:
//   go run scripts/bench/main.go                     # localhost
//   go run scripts/bench/main.go -url https://staging.cambia.jasonyu.io
//   go run scripts/bench/main.go -url http://100.64.1.2  # tailscale IP
//   go run scripts/bench/main.go -n 200              # 200 WS round trips
package main

import (
	"crypto/tls"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"sort"
	"strings"
	"time"

	"golang.org/x/net/websocket"
)

func main() {
	baseURL := flag.String("url", "http://localhost", "Base URL of the Cambia server (http or https)")
	n := flag.Int("n", 100, "Number of WS round-trip iterations")
	flag.Parse()

	base := strings.TrimRight(*baseURL, "/")
	jar, _ := cookiejar.New(nil)
	client := &http.Client{
		Jar:     jar,
		Timeout: 10 * time.Second,
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
	}

	fmt.Printf("Benchmarking %s (%d WS iterations)\n\n", base, *n)

	// --- 0. Provision guest session (sets auth cookie) ---
	t0 := time.Now()
	resp, err := client.Get(base + "/user/guest")
	guestTime := time.Since(t0)
	if err != nil {
		fmt.Printf("FATAL: cannot reach server: %v\n", err)
		return
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		fmt.Printf("FATAL: /user/guest returned %d: %s\n", resp.StatusCode, string(body))
		return
	}
	fmt.Printf("Guest session:        %s  (status %d)\n\n", guestTime.Round(time.Microsecond), resp.StatusCode)

	// --- 1. HTTP: Authed /user/me RTT ---
	meTimes := bench(*n, func() error {
		resp, err := client.Get(base + "/user/me")
		if err != nil {
			return err
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 {
			return fmt.Errorf("status %d", resp.StatusCode)
		}
		return nil
	})
	printStats("HTTP GET /user/me", meTimes)

	// --- 2. HTTP: Create lobby ---
	lobbyTimes := make([]time.Duration, 0, 5)
	var lobbyID string
	for i := 0; i < 5; i++ {
		t0 := time.Now()
		resp, err := client.Post(base+"/lobby/create", "application/json", strings.NewReader(`{"type":"public","gameMode":"head_to_head"}`))
		d := time.Since(t0)
		if err != nil {
			fmt.Printf("  lobby create error: %v\n", err)
			continue
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 {
			fmt.Printf("  lobby create status %d: %s\n", resp.StatusCode, string(body))
			continue
		}
		lobbyTimes = append(lobbyTimes, d)

		var lob struct {
			ID string `json:"id"`
		}
		json.Unmarshal(body, &lob)
		if lob.ID != "" {
			lobbyID = lob.ID
		}
	}
	printStats("HTTP POST /lobby/create", lobbyTimes)

	if lobbyID == "" {
		fmt.Println("FATAL: could not create lobby")
		return
	}
	fmt.Printf("  Using lobby: %s\n\n", lobbyID)

	// --- 3. WebSocket: Connect ---
	wsScheme := "ws"
	if strings.HasPrefix(base, "https") {
		wsScheme = "wss"
	}
	host := strings.TrimPrefix(strings.TrimPrefix(base, "https://"), "http://")
	wsURL := fmt.Sprintf("%s://%s/ws/%s", wsScheme, host, lobbyID)

	// Extract cookies for WS handshake
	u, _ := url.Parse(base)
	cookies := jar.Cookies(u)
	cookieHeader := ""
	for _, c := range cookies {
		if cookieHeader != "" {
			cookieHeader += "; "
		}
		cookieHeader += c.Name + "=" + c.Value
	}

	origin := base
	config, err := websocket.NewConfig(wsURL, origin)
	if err != nil {
		fmt.Printf("FATAL: ws config: %v\n", err)
		return
	}
	config.Header.Set("Cookie", cookieHeader)
	config.TlsConfig = &tls.Config{InsecureSkipVerify: true}

	t0 = time.Now()
	ws, err := websocket.DialConfig(config)
	wsConnectTime := time.Since(t0)
	if err != nil {
		fmt.Printf("FATAL: ws connect: %v\n", err)
		return
	}
	defer ws.Close()
	fmt.Printf("WS Connect:           %s\n\n", wsConnectTime.Round(time.Microsecond))

	// Drain initial lobby_state message
	drainOne(ws)

	// --- 4. WebSocket: Message RTT (chat messages) ---
	wsTimes := make([]time.Duration, 0, *n)
	for i := 0; i < *n; i++ {
		msg := fmt.Sprintf(`{"type":"chat","last_seq":0,"body":{"msg":"bench-%d"}}`, i)
		t0 := time.Now()
		if _, err := ws.Write([]byte(msg)); err != nil {
			fmt.Printf("  ws write error at iter %d: %v\n", i, err)
			break
		}
		buf := make([]byte, 4096)
		ws.SetReadDeadline(time.Now().Add(5 * time.Second))
		_, err := ws.Read(buf)
		d := time.Since(t0)
		if err != nil {
			fmt.Printf("  ws read error at iter %d: %v\n", i, err)
			break
		}
		wsTimes = append(wsTimes, d)
	}
	printStats("WS RTT (chat msg)", wsTimes)

	// --- 5. WebSocket: Ready/unready RTT ---
	readyTimes := make([]time.Duration, 0, *n)
	for i := 0; i < *n; i++ {
		msg := `{"type":"ready","last_seq":0}`
		if i%2 == 1 {
			msg = `{"type":"unready","last_seq":0}`
		}
		t0 := time.Now()
		ws.Write([]byte(msg))
		buf := make([]byte, 8192)
		ws.SetReadDeadline(time.Now().Add(5 * time.Second))
		_, err := ws.Read(buf)
		d := time.Since(t0)
		if err != nil {
			break
		}
		readyTimes = append(readyTimes, d)
	}
	printStats("WS RTT (ready toggle)", readyTimes)

	// --- 6. HTTP: matchmaking/queues ---
	queueTimes := bench(*n, func() error {
		resp, err := client.Get(base + "/matchmaking/queues")
		if err != nil {
			return err
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		return nil
	})
	printStats("HTTP GET /matchmaking/queues", queueTimes)
}

func bench(n int, fn func() error) []time.Duration {
	times := make([]time.Duration, 0, n)
	for i := 0; i < n; i++ {
		t0 := time.Now()
		if err := fn(); err != nil {
			break
		}
		times = append(times, time.Since(t0))
	}
	return times
}

func drainOne(ws *websocket.Conn) {
	buf := make([]byte, 8192)
	ws.SetReadDeadline(time.Now().Add(3 * time.Second))
	ws.Read(buf)
}

func printStats(label string, times []time.Duration) {
	if len(times) == 0 {
		fmt.Printf("%-28s  no data\n\n", label+":")
		return
	}
	sort.Slice(times, func(i, j int) bool { return times[i] < times[j] })
	total := time.Duration(0)
	for _, t := range times {
		total += t
	}
	avg := total / time.Duration(len(times))
	p50 := times[len(times)*50/100]
	p95 := times[len(times)*95/100]
	p99 := times[len(times)*99/100]
	min := times[0]
	max := times[len(times)-1]

	fmt.Printf("%-28s  n=%-4d  min=%-8s  p50=%-8s  p95=%-8s  p99=%-8s  max=%-8s  avg=%s\n",
		label+":", len(times),
		min.Round(time.Microsecond), p50.Round(time.Microsecond),
		p95.Round(time.Microsecond), p99.Round(time.Microsecond),
		max.Round(time.Microsecond), avg.Round(time.Microsecond))
	fmt.Println()
}
