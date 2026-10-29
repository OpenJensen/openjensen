import { publicPath } from './base-path';
import {
  affectedLink,
  so101Camera,
  descriptionForRole,
  visualMatrix,
  type Camera,
  type RobotPose,
} from './so101-schematic';
import type { RobotJoint, RobotRole } from './robot-api';

export type Mesh = { vertices: Float32Array; count: number };
export function parseBinaryStl(buffer: ArrayBuffer): Mesh {
  if (buffer.byteLength < 84) throw new Error('Incomplete CAD mesh.');
  const data = new DataView(buffer),
    triangles = data.getUint32(80, true);
  if (!triangles || triangles > 1_000_000 || buffer.byteLength !== 84 + triangles * 50)
    throw new Error('Invalid CAD mesh.');
  const vertices = new Float32Array(triangles * 18);
  for (let face = 0; face < triangles; face++) {
    const offset = 84 + face * 50,
      normal = [data.getFloat32(offset, true), data.getFloat32(offset + 4, true), data.getFloat32(offset + 8, true)];
    for (let v = 0; v < 3; v++)
      for (let k = 0; k < 3; k++) {
        const coordinate = data.getFloat32(offset + 12 + v * 12 + k * 4, true);
        if (!Number.isFinite(coordinate) || !Number.isFinite(normal[k])) throw new Error('Non-finite CAD vertex.');
        vertices[face * 18 + v * 6 + k] = coordinate;
        vertices[face * 18 + v * 6 + 3 + k] = normal[k];
      }
  }
  return { vertices, count: triangles * 3 };
}
const cachedMeshes = new Map<string, Promise<Mesh>>();
export async function loadSo101Meshes(role: RobotRole = 'follower'): Promise<Map<string, Mesh>> {
  const entries = await Promise.all(
    [...new Set(descriptionForRole(role).visuals.map((visual) => visual.mesh))].map(async (mesh) => {
      let pending = cachedMeshes.get(mesh);
      if (!pending) {
        pending = fetch(publicPath(`/robots/so101/assets/${mesh}`), { cache: 'force-cache' })
          .then(async (response) => {
            if (!response.ok) throw new Error('CAD mesh unavailable.');
            return parseBinaryStl(await response.arrayBuffer());
          })
          .catch((error) => {
            cachedMeshes.delete(mesh);
            throw error;
          });
        cachedMeshes.set(mesh, pending);
      }
      return [mesh, await pending] as const;
    }),
  );
  return new Map(entries);
}
const vertexSource = `attribute vec3 position;attribute vec3 normal;uniform mat4 model;uniform mat4 camera;varying vec3 worldNormal;void main(){worldNormal=mat3(model)*normal;gl_Position=camera*model*vec4(position,1.0);}`;
const fragmentSource = `precision mediump float;uniform vec3 color;varying vec3 worldNormal;void main(){vec3 n=normalize(worldNormal);float diffuse=max(dot(n,normalize(vec3(0.4,-0.65,0.8))),0.0);float fill=max(dot(n,normalize(vec3(-0.7,0.2,0.4))),0.0);gl_FragColor=vec4(color*(0.48+0.43*diffuse+0.15*fill),1.0);}`;
function shader(gl: WebGLRenderingContext, type: number, source: string) {
  const value = gl.createShader(type);
  if (!value) throw new Error('WebGL shader unavailable.');
  gl.shaderSource(value, source);
  gl.compileShader(value);
  if (!gl.getShaderParameter(value, gl.COMPILE_STATUS)) {
    gl.deleteShader(value);
    throw new Error('WebGL shader unavailable.');
  }
  return value;
}
function rgb(value: string): [number, number, number] {
  const hex = value.trim().replace('#', '');
  if (/^[0-9a-f]{6}$/i.test(hex))
    return [
      parseInt(hex.slice(0, 2), 16) / 255,
      parseInt(hex.slice(2, 4), 16) / 255,
      parseInt(hex.slice(4, 6), 16) / 255,
    ];
  const parts = value.match(/[\d.]+/g);
  return parts && parts.length >= 3
    ? [Number(parts[0]) / 255, Number(parts[1]) / 255, Number(parts[2]) / 255]
    : [0.2, 0.4, 0.6];
}
export type CadRenderer = {
  render: (pose: RobotPose, selected: RobotJoint, view?: Camera) => void;
  dispose: () => void;
};
export function createSo101Renderer(
  canvas: HTMLCanvasElement,
  meshes: Map<string, Mesh>,
  role: RobotRole = 'follower',
): CadRenderer {
  const description = descriptionForRole(role);
  const gl = canvas.getContext('webgl', { antialias: true, alpha: false, preserveDrawingBuffer: true });
  if (!gl) throw new Error('WebGL unavailable.');
  const vertex = shader(gl, gl.VERTEX_SHADER, vertexSource);
  let fragment: WebGLShader;
  try {
    fragment = shader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  } catch (error) {
    gl.deleteShader(vertex);
    throw error;
  }
  const program = gl.createProgram();
  if (!program) {
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    throw new Error('WebGL unavailable.');
  }
  const buffers = new Map<string, { buffer: WebGLBuffer; count: number }>();
  try {
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error('WebGL unavailable.');
    for (const [name, mesh] of meshes) {
      const buffer = gl.createBuffer();
      if (!buffer) throw new Error('WebGL buffer unavailable.');
      buffers.set(name, { buffer, count: mesh.count });
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, mesh.vertices, gl.STATIC_DRAW);
      if (gl.getError() !== gl.NO_ERROR) throw new Error('CAD buffer allocation unavailable.');
    }
  } catch (error) {
    for (const entry of buffers.values()) gl.deleteBuffer(entry.buffer);
    gl.deleteProgram(program);
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    throw error;
  }
  const pos = gl.getAttribLocation(program, 'position'),
    normal = gl.getAttribLocation(program, 'normal'),
    model = gl.getUniformLocation(program, 'model'),
    camera = gl.getUniformLocation(program, 'camera'),
    color = gl.getUniformLocation(program, 'color');
  let disposed = false;
  return {
    render(pose, selected, view = so101Camera) {
      if (disposed || gl.isContextLost()) return;
      const ratio = Math.min(window.devicePixelRatio || 1, 1.5),
        width = Math.max(1, Math.round(canvas.clientWidth * ratio)),
        height = Math.max(1, Math.round(canvas.clientHeight * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      const style = getComputedStyle(document.documentElement),
        background = rgb(style.getPropertyValue('--subtle')),
        accent = rgb(style.getPropertyValue('--accent')),
        dark = document.documentElement.dataset.theme === 'dark';
      gl.viewport(0, 0, width, height);
      gl.clearColor(...background, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      gl.useProgram(program);
      gl.uniformMatrix4fv(camera, false, new Float32Array(view.matrix));
      gl.enableVertexAttribArray(pos);
      gl.enableVertexAttribArray(normal);
      const child = description.joints.find((joint) => joint.name === selected)!.child;
      for (const visual of description.visuals) {
        const mesh = buffers.get(visual.mesh)!;
        gl.bindBuffer(gl.ARRAY_BUFFER, mesh.buffer);
        gl.vertexAttribPointer(pos, 3, gl.FLOAT, false, 24, 0);
        gl.vertexAttribPointer(normal, 3, gl.FLOAT, false, 24, 12);
        gl.uniformMatrix4fv(model, false, new Float32Array(visualMatrix(pose, visual)));
        const base =
          visual.material === 'sts3215'
            ? dark
              ? [0.48, 0.53, 0.59]
              : [0.27, 0.31, 0.37]
            : dark
              ? [0.64, 0.69, 0.75]
              : [0.83, 0.86, 0.9];
        const highlight = visual.link === child ? 0.85 : affectedLink(selected, visual.link, role) ? 0.25 : 0;
        gl.uniform3f(
          color,
          base[0] * (1 - highlight) + accent[0] * highlight,
          base[1] * (1 - highlight) + accent[1] * highlight,
          base[2] * (1 - highlight) + accent[2] * highlight,
        );
        gl.drawArrays(gl.TRIANGLES, 0, mesh.count);
      }
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      for (const entry of buffers.values()) gl.deleteBuffer(entry.buffer);
      gl.deleteProgram(program);
      gl.deleteShader(vertex);
      gl.deleteShader(fragment);
    },
  };
}
