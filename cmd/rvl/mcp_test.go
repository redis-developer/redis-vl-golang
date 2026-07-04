package main

import "testing"

func TestCheckHTTPBind(t *testing.T) {
	cases := []struct {
		name                 string
		host                 string
		authEnabled          bool
		allowUnauthenticated bool
		wantWarn             bool
		wantErr              bool
	}{
		{"loopback unauthenticated", "127.0.0.1", false, false, true, false},
		{"localhost unauthenticated", "localhost", false, false, true, false},
		{"ipv6 loopback unauthenticated", "::1", false, false, true, false},
		{"non-loopback unauthenticated refused", "0.0.0.0", false, false, false, true},
		{"non-loopback unauthenticated with override", "0.0.0.0", false, true, true, false},
		{"non-loopback with auth", "0.0.0.0", true, false, false, false},
		{"loopback with auth", "127.0.0.1", true, false, false, false},
		{"auth makes override irrelevant", "0.0.0.0", true, true, false, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			warn, err := checkHTTPBind(tc.host, tc.authEnabled, tc.allowUnauthenticated)
			if (err != nil) != tc.wantErr {
				t.Errorf("err = %v, wantErr = %v", err, tc.wantErr)
			}
			if warn != tc.wantWarn {
				t.Errorf("warn = %v, want %v", warn, tc.wantWarn)
			}
		})
	}
}
