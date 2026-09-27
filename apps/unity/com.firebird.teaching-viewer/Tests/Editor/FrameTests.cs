using System;
using System.IO;
using System.Text;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace Firebird.TeachingViewer.Tests
{
    public sealed class FrameTests
    {
        [Test]
        public void ValidFramePreservesIdentityAndCopiesBuffers()
        {
            var frame = FrameCodec.Decode(Fixture.Bytes(), Fixture.Binding());
            Assert.That(frame.StateRad, Is.EqualTo(new[] { 0.25, 0.5 }));
            var copy = frame.CopyRgb(); copy[0] = 255;
            Assert.That(frame.CopyRgb()[0], Is.EqualTo(17));
            Assert.That(frame.SessionId, Is.EqualTo("fixture-session"));
        }
        [TestCase("session")]
        [TestCase("camera")]
        [TestCase("context")]
        [TestCase("active")]
        [TestCase("joint_name")]
        [TestCase("joint_count")]
        [TestCase("range")]
        [TestCase("nan")]
        [TestCase("units")]
        [TestCase("bool_step")]
        [TestCase("fraction_revision")]
        [TestCase("dimension")]
        [TestCase("rgb")]
        [TestCase("hash")]
        [TestCase("extra")]
        [TestCase("fault")]
        [TestCase("clock")]
        [TestCase("negative_age")]
        public void RejectMalformedOrUnboundObservation(string damage)
        {
            var x = Fixture.Frame();
            switch (damage)
            {
                case "session": x["session_id"] = "other"; break;
                case "camera": x["camera_key"] = "other"; break;
                case "context": x["current_context"]["revision"] = 2; break;
                case "active": x["active_episode_id"] = "other"; break;
                case "joint_name": x["joints"][1] = "shoulder_pan"; break;
                case "joint_count": x["state_rad"] = new JArray(0); break;
                case "range": x["state_rad"][0] = 4; break;
                case "nan": x["state_rad"][0] = double.NaN; break;
                case "units": x["units"] = "deg"; break;
                case "bool_step": x["step"] = true; break;
                case "fraction_revision": x["revision"] = 1.2; break;
                case "dimension": x["width"] = 1920; break;
                case "rgb": x["rgb_base64"] = "AAAA"; break;
                case "hash": x["rgb_sha256"] = new string('0', 64); break;
                case "extra": x["run_command"] = "forbidden"; break;
                case "fault": x["mode"] = "faulted"; x["current_context"]["mode"] = "faulted"; break;
                case "clock": x["published_monotonic_ns"] = 99; break;
                case "negative_age": x["source_age_ns"] = -1; break;
            }
            Assert.Throws<InvalidDataException>(() => FrameCodec.Decode(Fixture.Bytes(x), Fixture.Binding()));
        }
        [Test]
        public void RejectDuplicateTrailingOversizedAndNonUtf8Json()
        {
            string valid = Encoding.UTF8.GetString(Fixture.Bytes());
            foreach (byte[] raw in new[]{Encoding.UTF8.GetBytes(valid.Replace("\"schema_version\":1", "\"schema_version\":1,\"schema_version\":1")),
                Encoding.UTF8.GetBytes(valid+"{}"),Encoding.Unicode.GetBytes(valid),new byte[Fixture.Binding().MaxJsonBytes+1]})
                Assert.Throws<InvalidDataException>(() => FrameCodec.Decode(raw, Fixture.Binding()));
        }
        [Test]
        public void RepeatedFrameCannotReviveStaleObservation()
        {
            var stream = new ObservationStream(Fixture.Binding());
            Assert.That(stream.Accept(Fixture.Bytes(), 0, out var changed), Is.True); Assert.That(changed, Is.True);
            Assert.That(stream.Accept(Fixture.Bytes(), 1, out changed), Is.True); Assert.That(changed, Is.False);
            Assert.That(stream.Accept(Fixture.Bytes(), 3, out changed), Is.False);
            Assert.That(stream.IsFresh(3), Is.False);
            var next = Fixture.Frame(); next["step"] = 1; next["sim_time"] = 1.0 / 30; next["observation_received_monotonic_ns"] = 300; next["published_monotonic_ns"] = 400;
            Assert.That(stream.Accept(Fixture.Bytes(next), 3, out changed), Is.True); Assert.That(changed, Is.True);
        }
        [TestCase("same_id_changed")]
        [TestCase("reordered")]
        [TestCase("same_step_new_time")]
        [TestCase("old_context")]
        public void InvalidUpdateDoesNotReplaceLastCapture(string damage)
        {
            var stream = new ObservationStream(Fixture.Binding()); stream.Accept(Fixture.Bytes(), 0, out _);
            var first = stream.Latest; var x = Fixture.Frame();
            if (damage == "same_id_changed") x["state_rad"][0] = 0.5;
            if (damage == "reordered") x["observation_received_monotonic_ns"] = 99;
            if (damage == "same_step_new_time") { x["observation_received_monotonic_ns"] = 300; x["published_monotonic_ns"] = 400; }
            if (damage == "old_context") x["current_context"]["revision"] = 2;
            Assert.That(stream.Accept(Fixture.Bytes(x), 1, out _), Is.False);
            Assert.That(stream.Latest, Is.SameAs(first)); Assert.That(stream.IsFresh(1), Is.False);
            stream.Disconnect(); Assert.That(stream.Status, Is.EqualTo("Disconnected"));
        }
        [Test]
        public void StaleSourceIsNotPresentedAsFreshEvenOnFirstReceipt()
        {
            var x = Fixture.Frame(); x["source_age_ns"] = 3_000_000_000L;
            var stream = new ObservationStream(Fixture.Binding());
            Assert.That(stream.Accept(Fixture.Bytes(x), 0, out _), Is.False);
        }
        [Test]
        public void StaleNewCaptureDoesNotRelabelLastAcceptedPixels()
        {
            var stream = new ObservationStream(Fixture.Binding());
            Assert.That(stream.Accept(Fixture.Bytes(), 0, out _), Is.True);
            var original = stream.Latest;
            var stale = Fixture.Frame(); stale["step"] = 1; stale["source_age_ns"] = 3_000_000_000L;
            stale["observation_received_monotonic_ns"] = 300; stale["published_monotonic_ns"] = 400;
            Assert.That(stream.Accept(Fixture.Bytes(stale), 0.1, out _), Is.False);
            Assert.That(stream.Latest, Is.SameAs(original));
        }
        [Test]
        public void SourceAgeIncludesRequestDurationConservatively()
        {
            var x = Fixture.Frame(); x["source_age_ns"] = 1_500_000_000L;
            var stream = new ObservationStream(Fixture.Binding());
            Assert.That(stream.Accept(Fixture.Bytes(x), 10, out _, 0.7), Is.False);
        }
        [Test]
        public void SameCaptureRepublishPreservesSourceAgeWithoutRejectingNewPublicationTime()
        {
            var stream = new ObservationStream(Fixture.Binding());
            Assert.That(stream.Accept(Fixture.Bytes(), 0, out _), Is.True);
            var repeated = Fixture.Frame(); repeated["published_monotonic_ns"] = 300; repeated["source_age_ns"] = 200;
            Assert.That(stream.Accept(Fixture.Bytes(repeated), 0.1, out var changed), Is.True);
            Assert.That(changed, Is.False);
            Assert.That(stream.Accept(Fixture.Bytes(), 0.2, out _), Is.False, "Older publication must not replace newer status.");
        }
        [Test]
        public void RejectedContextRetiresCaptureUntilANewerObservationArrives()
        {
            var stream = new ObservationStream(Fixture.Binding());
            Assert.That(stream.Accept(Fixture.Bytes(), 0, out _), Is.True);
            var superseded = Fixture.Frame(); superseded["current_context"]["revision"] = 2;
            Assert.That(stream.Accept(Fixture.Bytes(superseded), 0.1, out _), Is.False);
            Assert.That(stream.Accept(Fixture.Bytes(), 0.2, out _), Is.False);
            var fresh = Fixture.Frame(); fresh["revision"] = 2; fresh["current_context"]["revision"] = 2;
            fresh["step"] = 1; fresh["observation_received_monotonic_ns"] = 300; fresh["published_monotonic_ns"] = 400;
            Assert.That(stream.Accept(Fixture.Bytes(fresh), 0.3, out var changed), Is.True);
            Assert.That(changed, Is.True);
        }
        [Test]
        public void NewlineSuffixCannotBypassIdentifierAnchors()
        {
            Assert.Throws<ArgumentException>(() => new JointLimit("joint\n", -1, 1));
            Assert.Throws<ArgumentException>(() => new ViewBinding("session\n", "camera", "/World/Camera", 4, 2, new[] { new JointLimit("joint", -1, 1) }));
            Assert.Throws<ArgumentException>(() => new ViewBinding("session", "camera", "/World/Camera\n", 4, 2, new[] { new JointLimit("joint", -1, 1) }));
            var frame = Fixture.Frame(); frame["episode_id"] = "episode-one\n";
            Assert.Throws<InvalidDataException>(() => FrameCodec.Decode(Fixture.Bytes(frame), Fixture.Binding()));
        }
        [Test]
        public void GenuineWorkerGeneratedSnapshotMatchesTheUnityContract()
        {
            string path = Environment.GetEnvironmentVariable("FIREBIRD_UNITY_FRAME_FIXTURE");
            if (string.IsNullOrEmpty(path)) Assert.Ignore("Explicit generated worker fixture path required.");
            byte[] raw = File.ReadAllBytes(path);
            string expectedSession = Environment.GetEnvironmentVariable("FIREBIRD_UNITY_EXPECTED_SESSION");
            Assert.That(expectedSession, Is.Not.Null.And.Not.Empty, "Expected session must be independently configured.");
            var binding = new ViewBinding(expectedSession, "observation.images.front", "/World/Camera", 32, 32,
                new[] { new JointLimit("joint_0", -1, 1), new JointLimit("joint_1", -1, 1), new JointLimit("joint_2", -1, 1), new JointLimit("joint_3", -1, 1), new JointLimit("joint_4", -1, 1), new JointLimit("joint_5", -1, 1) }, true);
            var result = FrameCodec.Decode(raw, binding);
            Assert.That(result.Mode, Is.EqualTo("idle")); Assert.That(result.ActiveEpisodeId, Is.Null);
            Assert.That(result.CopyRgb().Length, Is.EqualTo(32 * 32 * 3)); Assert.That(result.StateRad, Is.All.EqualTo(0));
        }
        [TestCase("https://127.0.0.1:8000")]
        [TestCase("http://localhost:8000")]
        [TestCase("http://127.0.0.1")]
        [TestCase("http://127.0.0.1:8000/frame")]
        [TestCase("http://127.0.0.1:8000/?token=x")]
        [TestCase("http://secret@127.0.0.1:8000")]
        [TestCase("http://127.0.0.1:8000/#x")]
        [TestCase("http://2130706433:8000")]
        public void RejectRemoteOrAmbiguousOrigin(string origin)
        { Assert.Throws<ArgumentException>(() => TeachingViewer.ValidateOrigin(origin)); }
    }
}
