using System;
using System.Collections;
using System.IO;
using System.Text;
using UnityEngine;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace Firebird.TeachingViewer
{
    /// <summary>Camera and numeric joint display only. No motor, physics, command, or room API.</summary>
    public sealed class TeachingViewer : MonoBehaviour
    {
        ViewBinding binding;
        ObservationStream stream;
        string origin;
        string token; // runtime only: never serialized into a scene, Inspector, or log
        Coroutine polling;
        CancellationTokenSource stop;
        Texture2D texture;
        string jointText = "No observation received.";
        double roundTripMs;
        public Texture2D CameraTexture => texture;
        public ObservationStream Stream => stream;
        public bool IsPolling => polling != null;
        public static Uri ValidateOrigin(string value)
        {
            if (!Uri.TryCreate(value, UriKind.Absolute, out var uri) || uri.Scheme != "http" ||
                uri.Host != "127.0.0.1" || uri.Port <= 0 || uri.Port > 65535 ||
                !string.IsNullOrEmpty(uri.UserInfo) || !string.IsNullOrEmpty(uri.Query) ||
                !string.IsNullOrEmpty(uri.Fragment) || uri.AbsolutePath != "/")
                throw new ArgumentException("Viewer requires an explicit http://127.0.0.1:PORT origin.");
            // Uri normalizes aliases; only the literal loopback spelling is accepted.
            if (!value.StartsWith("http://127.0.0.1:", StringComparison.Ordinal))
                throw new ArgumentException("Viewer requires an explicit literal loopback port.");
            return uri;
        }
        public void Configure(string loopbackOrigin, string bearerToken, ViewBinding viewBinding)
        {
            var uri = ValidateOrigin(loopbackOrigin);
            if (bearerToken == null || bearerToken.Length < 32 || bearerToken.Length > 4096 ||
                Array.Exists(bearerToken.ToCharArray(), c => c < 33 || c > 126))
                throw new ArgumentException("Invalid viewer credential.");
            if (viewBinding == null) throw new ArgumentNullException(nameof(viewBinding));
            StopViewing();
            binding = viewBinding; stream = new ObservationStream(binding);
            origin = uri.GetLeftPart(UriPartial.Authority); token = bearerToken;
            jointText = "No observation received.";
            if (texture != null) { Destroy(texture); texture = null; }
        }
        public void StartViewing()
        {
            if (!isActiveAndEnabled || binding == null || token == null)
                throw new InvalidOperationException("Configure an enabled viewer before explicitly starting it.");
            if (polling == null) { stop = new CancellationTokenSource(); polling = StartCoroutine(Poll(stop.Token)); }
        }
        public void StopViewing()
        {
            if (stop != null) { stop.Cancel(); stop.Dispose(); stop = null; }
            if (polling != null) { StopCoroutine(polling); polling = null; }
            stream?.Disconnect();
        }
        public bool Present(byte[] raw, double now, double requestElapsedSeconds = 0)
        {
            if (stream == null) return false;
            bool fresh = stream.Accept(raw, now, out bool changed, requestElapsedSeconds);
            if (!fresh || !changed) return fresh;
            if (texture == null) texture = new Texture2D(binding.Width, binding.Height, TextureFormat.RGB24, false, false);
            texture.LoadRawTextureData(stream.Latest.CopyRgb());
            texture.Apply(false, false);
            var text = new StringBuilder();
            for (int i = 0; i < binding.Joints.Count; ++i)
                text.Append(binding.Joints[i].Name).Append(": ").Append(stream.Latest.StateRad[i].ToString("F4", System.Globalization.CultureInfo.InvariantCulture)).Append(" rad\n");
            jointText = text.ToString();
            return true;
        }
        IEnumerator Poll(CancellationToken cancellation)
        {
            var pause = new WaitForSecondsRealtime(0.1f);
            while (!cancellation.IsCancellationRequested)
            {
                double started = Time.realtimeSinceStartupAsDouble;
                var request = ReadFrame(origin, token, binding.MaxJsonBytes, cancellation);
                while (!request.IsCompleted) yield return null;
                roundTripMs = (Time.realtimeSinceStartupAsDouble - started) * 1000;
                if (request.Status == TaskStatus.RanToCompletion && request.Result != null)
                    Present(request.Result, Time.realtimeSinceStartupAsDouble, roundTripMs / 1000);
                else stream.Disconnect();
                yield return pause;
            }
        }
        static async Task<byte[]> ReadFrame(string endpoint, string credential, int limit, CancellationToken stop)
        {
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(stop);
            timeout.CancelAfter(TimeSpan.FromSeconds(3));
            using var handler = new HttpClientHandler { UseProxy = false, AllowAutoRedirect = false, AutomaticDecompression = DecompressionMethods.None };
            using var client = new HttpClient(handler);
            using var request = new HttpRequestMessage(HttpMethod.Get, endpoint + "/frame");
            request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", credential);
            try
            {
                using var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, timeout.Token).ConfigureAwait(false);
                if (response.StatusCode != HttpStatusCode.OK || response.Content.Headers.ContentLength > limit ||
                    response.Content.Headers.ContentType?.MediaType != "application/json") return null;
                using var input = await response.Content.ReadAsStreamAsync().ConfigureAwait(false);
                using var output = new MemoryStream();
                var buffer = new byte[16384];
                while (true)
                {
                    int count = await input.ReadAsync(buffer, 0, Math.Min(buffer.Length, limit + 1 - (int)output.Length), timeout.Token).ConfigureAwait(false);
                    if (count == 0) return output.ToArray();
                    if (output.Length + count > limit) return null;
                    output.Write(buffer, 0, count);
                }
            }
            catch (Exception error) when (error is HttpRequestException || error is IOException || error is OperationCanceledException)
            { return null; }
        }
        public string DisplayStatus(double now) => stream == null ? "Disconnected" :
            stream.IsFresh(now) ? "Live observation" : stream.Latest == null ? stream.Status :
            "STALE / DISCONNECTED — last accepted image";
        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(12, 12, Mathf.Min(Screen.width - 24, 720), Screen.height - 24));
            GUILayout.Label(binding?.GeneratedFixture == true ? "GENERATED FIXTURE: no simulator or robot acceptance" : "OPEN JENSEN: receive-only simulator viewer");
            GUILayout.Label(DisplayStatus(Time.realtimeSinceStartupAsDouble));
            if (stream?.Latest != null)
            {
                GUILayout.Label($"Source age: {stream.Age(Time.realtimeSinceStartupAsDouble):F2}s | HTTP round trip: {roundTripMs:F1}ms");
                GUILayout.Label($"Simulation time: {stream.Latest.SimTime:F3}s | Model latency: not measured");
            }
            if (texture != null) GUILayout.Label(texture, GUILayout.MaxWidth(640), GUILayout.MaxHeight(360));
            GUILayout.Label(jointText);
            GUILayout.EndArea();
        }
        void OnDisable() { StopViewing(); token = null; }
        void OnDestroy() { if (texture != null) Destroy(texture); }
    }

}
