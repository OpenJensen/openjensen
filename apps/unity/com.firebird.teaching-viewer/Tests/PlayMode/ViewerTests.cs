using System;
using System.Collections;
using System.Net;
using System.Net.Sockets;
using System.Threading.Tasks;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace Firebird.TeachingViewer.Tests
{
    public sealed class ViewerTests
    {
        [UnityTest]
        public IEnumerator NoAutoConnectOrMotionAndValidatedCameraOnly()
        {
            var obj = new GameObject("Generated fixture viewer");
            var viewer = obj.AddComponent<TeachingViewer>();
            obj.transform.localRotation = Quaternion.Euler(11, 22, 33); var rotation = obj.transform.localRotation;
            try
            {
                yield return null;
                Assert.That(viewer.IsPolling, Is.False); Assert.That(viewer.CameraTexture, Is.Null);
                Assert.That(viewer.Present(Fixture.Bytes(), 0), Is.False);
                viewer.Configure("http://127.0.0.1:45678", Fixture.Token, Fixture.Binding());
                Assert.That(viewer.IsPolling, Is.False);
                Assert.That(viewer.Present(Fixture.Bytes(), 0), Is.True);
                Assert.That(viewer.CameraTexture.width, Is.EqualTo(4));
                Assert.That(viewer.CameraTexture.GetRawTextureData<byte>()[0], Is.EqualTo(17));
                Assert.That(obj.transform.localRotation, Is.EqualTo(rotation));
                var originalTexture = viewer.CameraTexture;
                var originalCapture = viewer.Stream.Latest;
                var invalidated = Fixture.Frame(); invalidated["current_context"]["revision"] = 2;
                Assert.That(viewer.Present(Fixture.Bytes(invalidated), 0.1), Is.False);
                Assert.That(viewer.Present(Fixture.Bytes(), 0.2), Is.False);
                Assert.That(viewer.CameraTexture, Is.SameAs(originalTexture));
                Assert.That(viewer.Stream.Latest, Is.SameAs(originalCapture));
                Assert.That(viewer.DisplayStatus(0.2), Does.StartWith("STALE / DISCONNECTED"));
                Assert.That(viewer.Stream.IsFresh(3), Is.False);
                viewer.StopViewing(); Assert.That(viewer.Stream.IsFresh(0), Is.False);
            }
            finally { UnityEngine.Object.Destroy(obj); }
            yield return null;
        }
        [UnityTest]
        public IEnumerator RealLoopbackGetIsReadOnlyAndAuthenticated()
        {
            var reservation = new TcpListener(IPAddress.Loopback, 0); reservation.Start();
            int port = ((IPEndPoint)reservation.LocalEndpoint).Port; reservation.Stop();
            using var server = new HttpListener(); server.Prefixes.Add($"http://127.0.0.1:{port}/"); server.Start();
            var request = server.GetContextAsync();
            var obj = new GameObject("Generated fixture HTTP viewer"); var viewer = obj.AddComponent<TeachingViewer>();
            try
            {
                viewer.Configure($"http://127.0.0.1:{port}", Fixture.Token, Fixture.Binding()); viewer.StartViewing();
                double deadline = Time.realtimeSinceStartupAsDouble + 5;
                while (!request.IsCompleted && Time.realtimeSinceStartupAsDouble < deadline) yield return null;
                Assert.That(request.IsCompleted, Is.True, "Viewer did not issue a GET within its bound.");
                var context = request.Result;
                Assert.That(context.Request.HttpMethod, Is.EqualTo("GET"));
                Assert.That(context.Request.RawUrl, Is.EqualTo("/frame"));
                Assert.That(context.Request.Headers["Authorization"], Is.EqualTo("Bearer " + Fixture.Token));
                byte[] raw = Fixture.Bytes(); context.Response.ContentType = "application/json"; context.Response.ContentLength64 = raw.Length;
                context.Response.OutputStream.Write(raw, 0, raw.Length); context.Response.Close();
                while (viewer.CameraTexture == null && Time.realtimeSinceStartupAsDouble < deadline) yield return null;
                Assert.That(viewer.CameraTexture, Is.Not.Null);
                Assert.That(viewer.Stream.IsFresh(Time.realtimeSinceStartupAsDouble), Is.True);
                viewer.StopViewing(); Assert.That(viewer.IsPolling, Is.False);
            }
            finally { viewer.StopViewing(); server.Stop(); UnityEngine.Object.Destroy(obj); }
            yield return null;
        }
    }
}
