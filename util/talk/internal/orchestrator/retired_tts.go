package orchestrator

import (
	"context"
	"fmt"
	"strings"
)

// A saved worker can still have the retired container on the shared TTS port.
// Retire it on replacement start, after preparation succeeded; prepare-only
// never stops a service. Current lifecycle images must first accept quiesce.
func (c *Controller) retireLegacyTTS(ctx context.Context, x Component) error {
	if !isManagedTTS(x) {
		return nil
	}
	host := c.host(x.Host)
	const name = "sparktalk-magpie-tts"
	out, err := executeHost(ctx, host, nil, "docker", "inspect", "--format", "{{.Config.Image}}|{{.State.Status}}", name)
	if err != nil {
		if isMissingContainer(err) {
			return nil
		}
		return fmt.Errorf("retired TTS inspection: %w", err)
	}
	fields := strings.Split(strings.TrimSpace(string(out)), "|")
	if len(fields) != 2 || !strings.HasPrefix(fields[0], "sparktalk-magpie-tts:") {
		return fmt.Errorf("%s is not a recognized retired Talk image", name)
	}
	if fields[1] == "running" {
		quiesced := strings.Contains(fields[0], "longform3-lifecycle")
		if quiesced {
			if err := c.idleWorkloadAction(ctx, x, "quiesce"); err != nil {
				return err
			}
		}
		if _, err := executeHost(ctx, host, nil, "docker", "stop", "-t", "30", name); err != nil {
			if quiesced {
				_ = c.idleWorkloadAction(ctx, x, "resume")
			}
			return err
		}
	}
	_, err = executeHost(ctx, host, nil, "docker", "rm", name)
	return err
}
