using System;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Firebird.TeachingViewer.Tests
{
    public static class Fixture
    {
        public const string Token = "generated-fixture-not-a-real-credential-123456";
        public static ViewBinding Binding() => new ViewBinding("fixture-session", "observation.images.front",
            "/World/Camera", 4, 2, new[] { new JointLimit("shoulder_pan", -1, 1), new JointLimit("gripper", 0, 1) }, true);
        public static JObject Frame()
        {
            byte[] rgb = Enumerable.Repeat((byte)17, 24).ToArray();
            using var sha = SHA256.Create();
            return new JObject
            {
                ["schema_version"] = 1,
                ["available"] = true,
                ["session_id"] = "fixture-session",
                ["revision"] = 1,
                ["active_episode_id"] = "fixture-episode",
                ["mode"] = "running",
                ["episode_id"] = "fixture-episode",
                ["step"] = 0,
                ["sim_time"] = 0.0,
                ["published_monotonic_ns"] = 200L,
                ["observation_received_monotonic_ns"] = 100L,
                ["source_age_ns"] = 100L,
                ["width"] = 4,
                ["height"] = 2,
                ["rgb_base64"] = Convert.ToBase64String(rgb),
                ["rgb_sha256"] = BitConverter.ToString(sha.ComputeHash(rgb)).Replace("-", "").ToLowerInvariant(),
                ["camera_key"] = "observation.images.front",
                ["camera_prim"] = "/World/Camera",
                ["joints"] = new JArray("shoulder_pan", "gripper"),
                ["state_rad"] = new JArray(0.25, 0.5),
                ["units"] = "rad",
                ["current_context"] = new JObject { ["session_id"] = "fixture-session", ["revision"] = 1, ["active_episode_id"] = "fixture-episode", ["mode"] = "running" }
            };
        }
        public static byte[] Bytes(JObject frame = null) => Encoding.UTF8.GetBytes((frame ?? Frame()).ToString(Formatting.None));
    }
}
