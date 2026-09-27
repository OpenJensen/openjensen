using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Firebird.TeachingViewer
{
    public sealed class JointLimit
    {
        public string Name { get; }
        public double Lower { get; }
        public double Upper { get; }
        public JointLimit(string name, double lower, double upper)
        {
            if (!Regex.IsMatch(name ?? "", @"^[A-Za-z_][A-Za-z0-9_]{0,95}\z") ||
                !FrameCodec.Finite(lower) || !FrameCodec.Finite(upper) || lower > upper)
                throw new ArgumentException("Invalid named joint limit.");
            Name = name; Lower = lower; Upper = upper;
        }
    }

    public sealed class ViewBinding
    {
        public string SessionId { get; }
        public string CameraKey { get; }
        public string CameraPrim { get; }
        public int Width { get; }
        public int Height { get; }
        public IReadOnlyList<JointLimit> Joints { get; }
        public double StaleAfterSeconds { get; }
        public int MaxJsonBytes => checked(4 * ((Width * Height * 3 + 2) / 3) + 65536);
        public bool GeneratedFixture { get; }
        public ViewBinding(string sessionId, string cameraKey, string cameraPrim,
            int width, int height, IEnumerable<JointLimit> joints, bool generatedFixture = false,
            double staleAfterSeconds = 2)
        {
            if (!FrameCodec.Identity(sessionId) || string.IsNullOrEmpty(cameraKey) ||
                cameraKey.Length > 256 || !Regex.IsMatch(cameraPrim ?? "", @"^(/[A-Za-z_][A-Za-z0-9_]*)+\z") ||
                width < 2 || width > 1920 || height < 2 || height > 1920 || width % 2 != 0 || height % 2 != 0 ||
                !FrameCodec.Finite(staleAfterSeconds) || staleAfterSeconds <= 0 || staleAfterSeconds > 5)
                throw new ArgumentException("Invalid explicit viewer binding.");
            var copy = joints?.ToArray();
            if (copy == null || copy.Length < 1 || copy.Length > 32 || copy.Any(j => j == null) ||
                copy.Select(j => j.Name).Distinct(StringComparer.Ordinal).Count() != copy.Length)
                throw new ArgumentException("Viewer needs distinct named joint limits.");
            SessionId = sessionId; CameraKey = cameraKey; CameraPrim = cameraPrim;
            Width = width; Height = height; Joints = Array.AsReadOnly(copy);
            StaleAfterSeconds = staleAfterSeconds; GeneratedFixture = generatedFixture;
        }
    }

    public sealed class Observation
    {
        public string SessionId { get; internal set; }
        public string EpisodeId { get; internal set; }
        public string ActiveEpisodeId { get; internal set; }
        public string Mode { get; internal set; }
        public long Revision { get; internal set; }
        public long Step { get; internal set; }
        public double SimTime { get; internal set; }
        public long ReceivedNs { get; internal set; }
        public long PublishedNs { get; internal set; }
        public double SourceAgeSeconds { get; internal set; }
        public string RgbSha256 { get; internal set; }
        internal byte[] Pixels;
        internal double[] Positions;
        public byte[] CopyRgb() => (byte[])Pixels.Clone();
        public IReadOnlyList<double> StateRad => Array.AsReadOnly(Positions);
        internal bool SameCapture(Observation other) => Revision == other.Revision &&
            EpisodeId == other.EpisodeId && ActiveEpisodeId == other.ActiveEpisodeId &&
            Mode == other.Mode && Step == other.Step && SimTime == other.SimTime &&
            RgbSha256 == other.RgbSha256 &&
            Positions.SequenceEqual(other.Positions);
    }

    public static class FrameCodec
    {
        static readonly HashSet<string> Fields = new HashSet<string>(new[] {
            "schema_version", "available", "session_id", "revision", "active_episode_id", "mode",
            "episode_id", "step", "sim_time", "published_monotonic_ns", "width", "height",
            "rgb_base64", "camera_key", "camera_prim", "rgb_sha256",
            "observation_received_monotonic_ns", "source_age_ns", "current_context",
            "joints", "state_rad", "units"
        }, StringComparer.Ordinal);
        public static bool Finite(double n) => !double.IsNaN(n) && !double.IsInfinity(n);
        internal static bool Identity(string s) => s != null && Regex.IsMatch(s, @"^[A-Za-z0-9_-]{1,96}\z");
        static InvalidDataException Bad() => new InvalidDataException("Observation does not match the bound session contract.");
        static string Text(JObject value, string key)
        {
            if (value[key]?.Type != JTokenType.String) throw Bad();
            return (string)value[key];
        }
        static string NullableIdentity(JObject value, string key)
        {
            if (value[key]?.Type == JTokenType.Null) return null;
            var result = Text(value, key);
            if (!Identity(result)) throw Bad();
            return result;
        }
        static long Integer(JObject value, string key, long min = 0)
        {
            if (value[key]?.Type != JTokenType.Integer) throw Bad();
            long n = value[key].Value<long>();
            if (n < min) throw Bad();
            return n;
        }
        static double Number(JToken value)
        {
            if (value == null || (value.Type != JTokenType.Float && value.Type != JTokenType.Integer)) throw Bad();
            double n = value.Value<double>();
            if (!Finite(n)) throw Bad();
            return n;
        }
        public static Observation Decode(byte[] raw, ViewBinding binding)
        {
            if (binding == null || raw == null || raw.Length == 0 || raw.Length > binding.MaxJsonBytes) throw Bad();
            try
            {
                using var text = new StringReader(new UTF8Encoding(false, true).GetString(raw));
                using var reader = new JsonTextReader(text) { MaxDepth = 8, DateParseHandling = DateParseHandling.None };
                var obj = JObject.Load(reader, new JsonLoadSettings { DuplicatePropertyNameHandling = DuplicatePropertyNameHandling.Error });
                if (reader.Read() || !Fields.SetEquals(obj.Properties().Select(p => p.Name))) throw Bad();
                if (obj["available"]?.Type != JTokenType.Boolean || obj["available"].Value<bool>() != true || Integer(obj, "schema_version") != 1 || Text(obj, "session_id") != binding.SessionId ||
                    Text(obj, "camera_key") != binding.CameraKey || Text(obj, "camera_prim") != binding.CameraPrim ||
                    Integer(obj, "width") != binding.Width || Integer(obj, "height") != binding.Height ||
                    Text(obj, "units") != "rad") throw Bad();
                var obs = new Observation
                {
                    SessionId = binding.SessionId,
                    EpisodeId = Text(obj, "episode_id"),
                    ActiveEpisodeId = NullableIdentity(obj, "active_episode_id"),
                    Mode = Text(obj, "mode"),
                    Revision = Integer(obj, "revision"),
                    Step = Integer(obj, "step"),
                    SimTime = Number(obj["sim_time"]),
                    ReceivedNs = Integer(obj, "observation_received_monotonic_ns"),
                    PublishedNs = Integer(obj, "published_monotonic_ns"),
                    SourceAgeSeconds = Integer(obj, "source_age_ns") / 1_000_000_000.0,
                    RgbSha256 = Text(obj, "rgb_sha256")
                };
                if (!Identity(obs.EpisodeId) || obs.SimTime < 0 || obs.PublishedNs < obs.ReceivedNs ||
                    Integer(obj, "source_age_ns") < obs.PublishedNs - obs.ReceivedNs ||
                    obs.Revision > 9007199254740991L || obs.Step > 9007199254740991L ||
                    (obs.ActiveEpisodeId == null ? obs.Mode != "idle" : obs.Mode == "idle") ||
                    (obs.Mode != "idle" && obs.Mode != "running" && obs.Mode != "paused") ||
                    (obs.ActiveEpisodeId != null ? obs.EpisodeId != obs.ActiveEpisodeId : !obs.EpisodeId.StartsWith("preview-", StringComparison.Ordinal))) throw Bad();
                var current = obj["current_context"] as JObject;
                if (current == null || !new HashSet<string>(new[] { "session_id", "revision", "active_episode_id", "mode" }).SetEquals(current.Properties().Select(p => p.Name)) ||
                    Text(current, "session_id") != obs.SessionId || Integer(current, "revision") != obs.Revision ||
                    NullableIdentity(current, "active_episode_id") != obs.ActiveEpisodeId || Text(current, "mode") != obs.Mode) throw Bad();
                var names = obj["joints"] as JArray;
                var positions = obj["state_rad"] as JArray;
                if (names == null || positions == null || names.Count != binding.Joints.Count || positions.Count != names.Count) throw Bad();
                obs.Positions = new double[names.Count];
                for (int i = 0; i < names.Count; ++i)
                {
                    var spec = binding.Joints[i];
                    if (names[i].Type != JTokenType.String || names[i].Value<string>() != spec.Name) throw Bad();
                    double q = Number(positions[i]);
                    if (q < spec.Lower || q > spec.Upper) throw Bad();
                    obs.Positions[i] = q;
                }
                string encoded = Text(obj, "rgb_base64");
                int expected = checked(binding.Width * binding.Height * 3);
                if (encoded.Length != 4 * ((expected + 2) / 3) || !Regex.IsMatch(obs.RgbSha256, @"^[a-f0-9]{64}\z")) throw Bad();
                obs.Pixels = Convert.FromBase64String(encoded);
                if (obs.Pixels.Length != expected || Convert.ToBase64String(obs.Pixels) != encoded) throw Bad();
                using var sha = SHA256.Create();
                string digest = BitConverter.ToString(sha.ComputeHash(obs.Pixels)).Replace("-", "").ToLowerInvariant();
                if (digest != obs.RgbSha256) throw Bad();
                return obs;
            }
            catch (Exception error) when (error is JsonException || error is FormatException || error is OverflowException || error is DecoderFallbackException || error is InvalidCastException)
            { throw Bad(); }
        }
    }

    public sealed class ObservationStream
    {
        readonly ViewBinding binding;
        double firstReceivedAt;
        double firstSourceAge;
        double lastNow;
        long lastPublishedNs;
        long retiredCaptureNs = -1;
        bool accepted;
        public Observation Latest { get; private set; }
        public string Status { get; private set; } = "Disconnected";
        public ObservationStream(ViewBinding binding) { this.binding = binding ?? throw new ArgumentNullException(nameof(binding)); }
        public bool Accept(byte[] raw, double now, out bool changed, double requestElapsedSeconds = 0)
        {
            changed = false;
            try
            {
                if (!FrameCodec.Finite(now) || now < lastNow || !FrameCodec.Finite(requestElapsedSeconds) || requestElapsedSeconds < 0) throw new InvalidDataException();
                lastNow = now;
                var incoming = FrameCodec.Decode(raw, binding);
                incoming.SourceAgeSeconds += requestElapsedSeconds;
                if (incoming.ReceivedNs <= retiredCaptureNs || incoming.SourceAgeSeconds > binding.StaleAfterSeconds) throw new InvalidDataException();
                if (Latest != null)
                {
                    if (incoming.ReceivedNs < Latest.ReceivedNs || incoming.PublishedNs < lastPublishedNs || incoming.Revision < Latest.Revision ||
                        (incoming.ReceivedNs == Latest.ReceivedNs && !incoming.SameCapture(Latest)) ||
                        (incoming.ReceivedNs > Latest.ReceivedNs && incoming.EpisodeId == Latest.EpisodeId && incoming.Step <= Latest.Step)) throw new InvalidDataException();
                }
                if (Latest == null || incoming.ReceivedNs > Latest.ReceivedNs)
                {
                    firstReceivedAt = now; firstSourceAge = incoming.SourceAgeSeconds;
                    Latest = incoming; changed = true;
                }
                lastPublishedNs = incoming.PublishedNs;
                firstSourceAge = Math.Max(firstSourceAge, incoming.SourceAgeSeconds - (now - firstReceivedAt));
                accepted = true;
                Status = Age(now) <= binding.StaleAfterSeconds ? "Live observation" : "Stale observation";
                return IsFresh(now);
            }
            catch (InvalidDataException)
            {
                if (Latest != null) retiredCaptureNs = Math.Max(retiredCaptureNs, Latest.ReceivedNs);
                accepted = false; Status = "Rejected observation"; return false;
            }
        }
        public double Age(double now) => Latest == null ? double.PositiveInfinity : firstSourceAge + Math.Max(0, now - firstReceivedAt);
        public bool IsFresh(double now) => accepted && FrameCodec.Finite(now) && now >= lastNow && Age(now) <= binding.StaleAfterSeconds;
        public void Disconnect() { accepted = false; Status = "Disconnected"; }
    }
}
