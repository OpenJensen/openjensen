# Teaching connections and camera preview

The application's API connection, teaching executor and optional voice service are separate connections. A ready Google Cloud account does not mean a VM or simulator is running. The Teaching page never starts one automatically.

An operator must start a reviewed teaching executor and configure `FIREBIRD_TEACHING_URL` plus `FIREBIRD_TEACHING_CONTROL_TOKEN_FILE` on the application host, following [the worker setup](../workers/teaching/README.md). These are operator settings; the browser cannot choose an arbitrary URL or token path. The existing relay only permits its authenticated loopback origin. Optional voice additionally needs `FIREBIRD_TEACHING_VOICE_URL` and the worker's isolated LiveKit/OpenRouter configuration. Camera viewing does not enable the microphone or join a room.

The page polls state every second. **Executor reachable** means a recently answered state request with a session ID, not proof of fresh rendering or progressing physics. A response older than five seconds disables controls; errors do likewise. Recording and correction continue to bind the fresh session/revision and require a final acknowledgement. Timeouts do not automatically retry commands or confirm motion.

## Atomic preview

The read-only `GET /api/v1/teaching/frame?session_id=...` relays the worker's [atomic schema 1 envelope](../workers/teaching/FRAME.md). The optional query binds the browser's selected session; direct callers without it still receive strict envelope validation. Legacy seven-field payloads are rejected because they cannot establish atomic identity. Unavailable responses retain `{schema_version:1,available:false}`.

The relay checks exact fields/version, current versus acquisition context, episode identity, camera metadata, finite matched native joints, pixel dimensions, base64 length and SHA256. It adds full application-to-executor request duration to the executor's source age and rejects observations older than five seconds. Source timestamps remain unchanged. Two application fields are additive: `relay_age_ns` is that conservative elapsed duration, and `capture_id` is the decimal source observation timestamp so JavaScript can compare the complete 64-bit identity without rounding.

The browser independently binds the frame to the displayed session/revision/episode/mode, validates pixels and their hash, and adds its complete request/decode duration. It never subtracts clocks on different hosts. Repeated identical responses cannot extend a capture's deadline. A stale, invalidated or disconnected capture stays visibly historical and cannot become fresh again by replay; a newer valid observation is required. New executor sessions reset this capture history. Frame transport has a 6-second deadline and 16 MiB body cap; the relay retains its 4-second deadline. Camera pixels and measured radians are shown together, independent of voice.

The preview does not issue simulator observations, steps, resets or actuator commands. Pausing or changing instruction may invalidate the previous frame until the executor produces another actual observation; the UI does not force rendering to make the image appear live. Joints are native measured radians, without claiming a calibrated physical mapping.

## Verification boundary

API tests use an authenticated generated loopback executor and the actual worker's envelope producer. Browser tests use explicitly generated camera pixels, verify rendered RGB and joints, disconnected/voice-independent behavior, corruption, staleness, replay and reconnect. These checks do not establish a live Isaac connection, trained-policy rollout, microphone/provider operation, cup pickup success or calibration. Those require separately authorized runtime checks with the actual scene and policy.
