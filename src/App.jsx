import { useState, useEffect, useRef, useCallback, useMemo } from "react";

// -- Complex number arithmetic (pure functions) --

const cAdd = (a, b) => ({ re: a.re + b.re, im: a.im + b.im });
const cMul = (a, b) => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re });
const cMagSq = (z) => z.re * z.re + z.im * z.im;
const cMag = (z) => Math.sqrt(cMagSq(z));

// -- Higher-order iterate function --
// Apply f to initial n times. This single function generalises
// Mandelbrot iteration, L-system rewriting, and more.
const iterate = (f, n, initial) => {
  let result = initial;
  for (let i = 0; i < n; i++) result = f(result);
  return result;
};

// -- Mandelbrot / Julia escape count (inlined arithmetic for performance) --
// Kept for the code panel display and as reference. GPU rendering uses GLSL below.

// -- WebGL shader sources --

const VERT_SRC = `attribute vec2 a_pos;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }`;

// const FRAG_SRC = `
// precision highp float;
// uniform vec2 u_resolution;
// uniform vec4 u_viewport; // left, top, width, height
// uniform int u_maxIter;
// uniform float u_bailout;
// uniform int u_paletteId;
// uniform int u_isJulia;
// uniform vec2 u_juliaC;

// vec3 hsl2rgb(float h, float s, float l) {
//   h = mod(h, 360.0); if (h < 0.0) h += 360.0;
//   float c = (1.0 - abs(2.0*l - 1.0)) * s;
//   float x = c * (1.0 - abs(mod(h/60.0, 2.0) - 1.0));
//   float m = l - c*0.5;
//   vec3 rgb;
//   if      (h < 60.0)  rgb = vec3(c,x,0);
//   else if (h < 120.0) rgb = vec3(x,c,0);
//   else if (h < 180.0) rgb = vec3(0,c,x);
//   else if (h < 240.0) rgb = vec3(0,x,c);
//   else if (h < 300.0) rgb = vec3(x,0,c);
//   else                 rgb = vec3(c,0,x);
//   return rgb + m;
// }

// vec3 palette(float t, int pid) {
//   if (t >= 1.0) return vec3(0.0);
//   if (pid == 0) { // classic
//     float a = t * 360.0 * 3.0;
//     return hsl2rgb(220.0+a, 0.85, 0.15+t*0.55);
//   } else if (pid == 1) { // fire
//     return vec3(min(1.0,t*3.0), min(1.0,max(0.0,t*3.0-1.0)), min(1.0,max(0.0,t*3.0-2.0)));
//   } else if (pid == 2) { // ocean
//     return hsl2rgb(200.0+t*60.0, 0.80, 0.10+t*0.50);
//   } else if (pid == 3) { // grayscale
//     return vec3(t);
//   } else if (pid == 4) { // rainbow
//     return hsl2rgb(t*360.0, 0.90, 0.50);
//   } else if (pid == 5) { // electric
//     return vec3(
//       sin(t*6.2832)*0.5+0.5,
//       sin(t*6.2832+2.094)*0.5+0.5,
//       sin(t*6.2832+4.189)*0.5+0.5
//     );
//   } else if (pid == 6) { // twilight
//     return hsl2rgb(260.0+t*100.0, 0.70, 0.10+t*0.45);
//   } else { // monochrome cyan
//     return vec3(0.0, t*0.863, t);
//   }
//   return vec3(t);
// }

// void main() {
//   vec2 uv = gl_FragCoord.xy / u_resolution;
//   float cx = u_viewport.x + uv.x * u_viewport.z;
//   float cy = u_viewport.y + (1.0 - uv.y) * u_viewport.w;
//   float zr, zi, cr, ci;
//   if (u_isJulia == 1) {
//     zr = cx; zi = cy;
//     cr = u_juliaC.x; ci = u_juliaC.y;
//   } else {
//     zr = 0.0; zi = 0.0;
//     cr = cx; ci = cy;
//   }
//   float b2 = u_bailout * u_bailout;
//   int iter = u_maxIter;
//   float zr2, zi2;
//   for (int i = 0; i < 10000; i++) {
//     if (i >= u_maxIter) break;
//     zr2 = zr*zr; zi2 = zi*zi;
//     if (zr2 + zi2 > b2) { iter = i; break; }
//     zi = 2.0*zr*zi + ci;
//     zr = zr2 - zi2 + cr;
//   }
//   if (iter >= u_maxIter) {
//     gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
//   } else {
//     float mag = sqrt(zr*zr + zi*zi);
//     float smooth_iter = float(iter) + 1.0 - log(log(mag)) / log(2.0);
//     float t = smooth_iter / float(u_maxIter);
//     t = clamp(t, 0.0, 0.9999);
//     vec3 col = palette(t, u_paletteId);
//     gl_FragColor = vec4(col, 1.0);
//   }
// }
// `;
// https://iquilezles.org/articles/palettes/ ?
//

const FRAG_SRC = `
precision highp float;
uniform vec2 u_resolution;
uniform vec4 u_viewport;
uniform int u_maxIter;
uniform float u_bailout;
uniform int u_paletteId;
uniform int u_isJulia;
uniform vec2 u_juliaC;
uniform int u_equation; 
uniform float u_cPower;
uniform float u_zPower;

vec3 hsl2rgb(float h, float s, float l) {
  h = mod(h, 360.0); if (h < 0.0) h += 360.0;
  float c = (1.0 - abs(2.0*l - 1.0)) * s;
  float x = c * (1.0 - abs(mod(h/60.0, 2.0) - 1.0));
  float m = l - c*0.5;
  vec3 rgb;
  if      (h < 60.0)  rgb = vec3(c,x,0);
  else if (h < 120.0) rgb = vec3(x,c,0);
  else if (h < 180.0) rgb = vec3(0,c,x);
  else if (h < 240.0) rgb = vec3(0,x,c);
  else if (h < 300.0) rgb = vec3(x,0,c);
  else                 rgb = vec3(c,0,x);
  return rgb + m;
}

// vec3 palette(float t, int pid) {
//   if (t >= 1.0) return vec3(0.0);
//   if (pid == 0) return hsl2rgb(220.0+t*360.0*3.0, 0.85, 0.15+t*0.55);
//   if (pid == 1) return vec3(min(1.0,t*3.0), min(1.0,max(0.0,t*3.0-1.0)), min(1.0,max(0.0,t*3.0-2.0)));
//   if (pid == 2) return hsl2rgb(200.0+t*60.0, 0.80, 0.10+t*0.50);
//   if (pid == 3) return vec3(t);
//   if (pid == 4) return hsl2rgb(t*360.0, 0.90, 0.50);
//   if (pid == 5) return vec3(sin(t*6.2832)*0.5+0.5, sin(t*6.2832+2.094)*0.5+0.5, sin(t*6.2832+4.189)*0.5+0.5);
//   if (pid == 6) return hsl2rgb(260.0+t*100.0, 0.70, 0.10+t*0.45);
//   return vec3(0.0, t*0.863, t);
// }

vec3 palette(float t, int pid) {
  if (t >= 1.0) return vec3(0.0);
  if (pid == 0) return hsl2rgb(220.0+t*360.0*3.0, 0.85, 0.15+t*0.55);
  if (pid == 1) return vec3(min(1.0,t*3.0), min(1.0,max(0.0,t*3.0-1.0)), min(1.0,max(0.0,t*3.0-2.0)));
  if (pid == 2) return hsl2rgb(200.0+t*60.0, 0.80, 0.10+t*0.50);
  if (pid == 3) return vec3(t);
  if (pid == 4) return hsl2rgb(t*360.0, 0.90, 0.50);
  if (pid == 5) return vec3(sin(t*6.2832)*0.5+0.5, sin(t*6.2832+2.094)*0.5+0.5, sin(t*6.2832+4.189)*0.5+0.5);
  if (pid == 6) return hsl2rgb(260.0+t*100.0, 0.70, 0.10+t*0.45);
  else if (pid == 8) { // Lava
    // Black - deep red - orange - bright yellow
    return vec3(
      min(1.0, t * 2.0),
      min(1.0, max(0.0, t * 2.0 - 0.75)),
      min(1.0, max(0.0, t * 3.0 - 2.5))
    );
  }

  else if (pid == 9) { // Neon Acid
    // Vibrant green-cyan-magenta cycling with high contrast
    return vec3(
      sin(t * 12.566 + 1.0) * 0.5 + 0.5,
      sin(t * 12.566 + 3.0) * 0.5 + 0.5,
      sin(t * 12.566 + 5.0) * 0.5 + 0.5
    );
  }

  else if (pid == 10) { // Frozen / Ice
    // White-blue cold palette
    return hsl2rgb(200.0 + t * 30.0, 0.60, 0.40 + t * 0.55);
  }

  else if (pid == 11) { // Vaporwave
    // Pink - cyan synthwave aesthetic
    return vec3(
      0.5 + 0.5 * sin(t * 6.2832 * 2.0 + 0.0),
      0.2 + 0.3 * sin(t * 6.2832 * 2.0 + 2.5),
      0.7 + 0.3 * sin(t * 6.2832 * 2.0 + 1.0)
    );
  }

  else if (pid == 12) { // Forest
    // Dark greens - warm golden browns
    return hsl2rgb(90.0 + t * 50.0, 0.65, 0.08 + t * 0.45);
  }

  else if (pid == 13) { // Cosine Gradient (Iq-style)
    // The famous Inigo Quilez cosine palette with custom parameters
    // palette(t) = a + b * cos(2π(c*t + d))
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.00, 0.33, 0.67);
    return a + b * cos(6.2832 * (c * t + d));
  }

  else if (pid == 14) { // Inferno (approximation of matplotlib's inferno)
    // Black - purple - red - orange - yellow
    return vec3(
      min(1.0, t * 1.5 + 0.1 * sin(t * 12.0)),
      max(0.0, t * t * 1.2 - 0.1),
      max(0.0, sin(t * 3.14159) * 0.8)
    );
  }

  else if (pid == 15) { // Gold / Treasure
    // Black - deep amber - bright gold - white
    float r = min(1.0, t * 2.5);
    float g = min(1.0, t * t * 2.0);
    float b_ch = min(1.0, max(0.0, t * t * t * 3.0 - 0.2));
    return vec3(r, g * 0.85, b_ch * 0.3);
  }

  else if (pid == 16) { // Plasma
    // Vibrant purple-pink-orange inspired by matplotlib's plasma
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 0.5);
    vec3 d = vec3(0.80, 0.90, 0.30);
    return a + b * cos(6.2832 * (c * t + d));
  }

  else if (pid == 17) { // Zebra / Binary Bands
    // Hard-edged black-and-white banding for structural analysis
    float bands = 20.0;
    float v = step(0.5, fract(t * bands));
    return vec3(v);
  }

  else if (pid == 18) { // Nebula
    // Deep space: black → purple → blue → pink → white
    return vec3(
      0.5 + 0.5 * cos(6.2832 * (t * 0.8 + 0.5)),
      0.2 * t + 0.1,
      0.5 + 0.5 * cos(6.2832 * (t * 0.6 + 0.0))
    );
  }
  return vec3(0.0, t*0.863, t); // fallback
}

void main() {
  vec2 uv = gl_FragCoord.xy / u_resolution;
  float cx = u_viewport.x + uv.x * u_viewport.z;
  float cy = u_viewport.y + (1.0 - uv.y) * u_viewport.w;
  float zr, zi, cr, ci;
  
  if (u_isJulia == 1) {
    zr = cx; zi = cy; cr = u_juliaC.x; ci = u_juliaC.y;
  } else {
    zr = 0.0; zi = 0.0; cr = cx; ci = cy;
  }
  
  if (u_cPower != 1.0) {
    float r = sqrt(cr*cr + ci*ci);
    if (r > 0.0) {
      float theta = atan(ci, cr);
      float rp = pow(r, u_cPower);
      cr = rp * cos(u_cPower * theta);
      ci = rp * sin(u_cPower * theta);
    } else { cr = 0.0; ci = 0.0; }
  }

  float b2 = u_bailout * u_bailout;
  int iter = u_maxIter;
  
  for (int i = 0; i < 10000; i++) {
    if (i >= u_maxIter) break;
    float r2 = zr*zr + zi*zi;
    if (r2 > b2) { iter = i; break; }
    
    float next_zr, next_zi;
    
    if (u_zPower != 2.0) {
      float tzr = zr, tzi = zi;
      if (u_equation == 1) { tzr = abs(zr); tzi = abs(zi); } // Burning Ship
      else if (u_equation == 2) { tzi = -zi; } // Tricorn
      
      float r = sqrt(tzr*tzr + tzi*tzi);
      float theta = atan(tzi, tzr);
      float rp = pow(r, u_zPower);
      next_zr = rp * cos(u_zPower * theta);
      next_zi = rp * sin(u_zPower * theta);
    } else {
      if (u_equation == 1) { next_zi = abs(2.0 * zr * zi); next_zr = zr*zr - zi*zi; }
      else if (u_equation == 2) { next_zi = -2.0 * zr * zi; next_zr = zr*zr - zi*zi; }
      else { next_zi = 2.0 * zr * zi; next_zr = zr*zr - zi*zi; }
    }
    
    zi = next_zi + ci;
    zr = next_zr + cr;
  }
  
  if (iter >= u_maxIter) {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
  } else {
    float mag = sqrt(zr*zr + zi*zi);
    float log_p = log(max(1.001, u_zPower));
    float smooth_iter = float(iter) + 1.0 - log(log(mag)) / log_p;
    float t = clamp(smooth_iter / float(u_maxIter), 0.0, 0.9999);
    gl_FragColor = vec4(palette(t, u_paletteId), 1.0);
  }
}
`;

// const PALETTE_ID_MAP = { classic: 0, fire: 1, ocean: 2, grayscale: 3, rainbow: 4, electric: 5, twilight: 6, monochrome: 7 };
const PALETTE_ID_MAP = { classic: 0, fire: 1, ocean: 2, grayscale: 3, rainbow: 4, electric: 5, twilight: 6, monochrome: 7, lava: 8, neon_acid: 9, frozen: 10, vaporwave: 11, forest: 12, cosine_gradient: 13, inferno: 14, gold: 15, plasma: 16, zebra: 17, nebula: 18 };

// Compile a shader, return null on failure
const compileShader = (gl, type, src) => {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(s));
    gl.deleteShader(s);
    return null;
  }
  return s;
};

// Build a WebGL program from vertex + fragment source
const buildProgram = (gl) => {
  const vs = compileShader(gl, gl.VERTEX_SHADER, VERT_SRC);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
  if (!vs || !fs) return null;
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(prog));
    return null;
  }
  return prog;
};

// Set up the fullscreen quad buffer
const setupQuad = (gl, prog) => {
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "a_pos");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
};


const mandelbrotEscape = (c, maxIter, bailout) => {
  let zr = 0, zi = 0;
  const b2 = bailout * bailout;
  for (let i = 0; i < maxIter; i++) {
    const zr2 = zr * zr;
    const zi2 = zi * zi;
    if (zr2 + zi2 > b2) {
      const mag = Math.sqrt(zr2 + zi2);
      const smooth = i + 1 - Math.log(Math.log(mag)) / Math.log(2);
      return { escaped: true, count: smooth, rawCount: i };
    }
    zi = 2 * zr * zi + c.im;
    zr = zr2 - zi2 + c.re;
  }
  return { escaped: false, count: maxIter, rawCount: maxIter };
};

const juliaEscape = (z0, c, maxIter, bailout) => {
  let zr = z0.re, zi = z0.im;
  const b2 = bailout * bailout;
  for (let i = 0; i < maxIter; i++) {
    const zr2 = zr * zr;
    const zi2 = zi * zi;
    if (zr2 + zi2 > b2) {
      const mag = Math.sqrt(zr2 + zi2);
      const smooth = i + 1 - Math.log(Math.log(mag)) / Math.log(2);
      return { escaped: true, count: smooth, rawCount: i };
    }
    zi = 2 * zr * zi + c.im;
    zr = zr2 - zi2 + c.re;
  }
  return { escaped: false, count: maxIter, rawCount: maxIter };
};

// -- grab Viewport aspect ratio correctly --
const getCorrectedViewport = (vp, canvasW, canvasH) => {
  const ch = vp.width * (canvasH / canvasW);
  return {
    left: vp.left,
    top: vp.top + vp.height / 2 - ch / 2,
    width: vp.width,
    height: ch
  };
};

// -- Coordinate mapping (pure) --

const pixelToComplex = (px, py, viewport, canvasW, canvasH) => ({
  re: viewport.left + (px / canvasW) * viewport.width,
  im: viewport.top + (py / canvasH) * viewport.height,
});

const complexToPixel = (c, viewport, canvasW, canvasH) => ({
  x: ((c.re - viewport.left) / viewport.width) * canvasW,
  y: ((c.im - viewport.top) / viewport.height) * canvasH,
});

// -- Colour palettes (pure functions: (t) => [r, g, b]) --

const hslToRgb = (h, s, l) => {
  h = h % 360;
  if (h < 0) h += 360;
  s /= 100; l /= 100;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs((h / 60) % 2 - 1));
  const m = l - c / 2;
  let r, g, b;
  if (h < 60) { r = c; g = x; b = 0; }
  else if (h < 120) { r = x; g = c; b = 0; }
  else if (h < 180) { r = 0; g = c; b = x; }
  else if (h < 240) { r = 0; g = x; b = c; }
  else if (h < 300) { r = x; g = 0; b = c; }
  else { r = c; g = 0; b = x; }
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
};

// const palettes = {
//   classic: {
//     name: "Classic",
//     fn: (t) => {
//       if (t >= 1) return [0, 0, 0];
//       const angle = t * 360 * 3;
//       return hslToRgb(220 + angle, 85, 15 + t * 55);
//     }
//   },
//   fire: {
//     name: "Fire",
//     fn: (t) => {
//       if (t >= 1) return [0, 0, 0];
//       const r = Math.min(255, Math.floor(t * 3 * 255));
//       const g = Math.min(255, Math.floor(Math.max(0, t * 3 - 1) * 255));
//       const b = Math.min(255, Math.floor(Math.max(0, t * 3 - 2) * 255));
//       return [r, g, b];
//     }
//   },
//   ocean: {
//     name: "Ocean",
//     fn: (t) => {
//       if (t >= 1) return [0, 0, 0];
//       return hslToRgb(200 + t * 60, 80, 10 + t * 50);
//     }
//   },
//   grayscale: {
//     name: "Grayscale",
//     fn: (t) => {
//       if (t >= 1) return [0, 0, 0];
//       const v = Math.floor(t * 255);
//       return [v, v, v];
//     }
//   },
//   rainbow: {
//     name: "Rainbow",
//     fn: (t) => {
//       if (t >= 1) return [0, 0, 0];
//       return hslToRgb(t * 360, 90, 50);
//     }
//   },
//   electric: {
//     name: "Electric",
//     fn: (t) => {
//       if (t >= 1) return [0, 0, 0];
//       const r = Math.floor(Math.sin(t * Math.PI * 2) * 127 + 128);
//       const g = Math.floor(Math.sin(t * Math.PI * 2 + 2.094) * 127 + 128);
//       const b = Math.floor(Math.sin(t * Math.PI * 2 + 4.189) * 127 + 128);
//       return [r, g, b];
//     }
//   },
//   twilight: {
//     name: "Twilight",
//     fn: (t) => {
//       if (t >= 1) return [0, 0, 0];
//       return hslToRgb(260 + t * 100, 70, 10 + t * 45);
//     }
//   },
//   monochrome: {
//     name: "Cyan Mono",
//     fn: (t) => {
//       if (t >= 1) return [0, 0, 0];
//       return [0, Math.floor(t * 220), Math.floor(t * 255)];
//     }
//   }
// };

const palettes = {
  classic: {
    name: "Classic",
    fn: (t) => {
      if (t >= 1) return [0, 0, 0];
      const angle = t * 360 * 3;
      return hslToRgb(220 + angle, 85, 15 + t * 55);
    }
  },
  fire: {
    name: "Fire",
    fn: (t) => {
      if (t >= 1) return [0, 0, 0];
      const r = Math.min(255, Math.floor(t * 3 * 255));
      const g = Math.min(255, Math.floor(Math.max(0, t * 3 - 1) * 255));
      const b = Math.min(255, Math.floor(Math.max(0, t * 3 - 2) * 255));
      return [r, g, b];
    }
  },
  ocean: {
    name: "Ocean",
    fn: (t) => {
      if (t >= 1) return [0, 0, 0];
      return hslToRgb(200 + t * 60, 80, 10 + t * 50);
    }
  },
  grayscale: {
    name: "Grayscale",
    fn: (t) => {
      if (t >= 1) return [0, 0, 0];
      const v = Math.floor(t * 255);
      return [v, v, v];
    }
  },
  rainbow: {
    name: "Rainbow",
    fn: (t) => {
      if (t >= 1) return [0, 0, 0];
      return hslToRgb(t * 360, 90, 50);
    }
  },
  electric: {
    name: "Electric",
    fn: (t) => {
      if (t >= 1) return [0, 0, 0];
      const r = Math.floor(Math.sin(t * Math.PI * 2) * 127 + 128);
      const g = Math.floor(Math.sin(t * Math.PI * 2 + 2.094) * 127 + 128);
      const b = Math.floor(Math.sin(t * Math.PI * 2 + 4.189) * 127 + 128);
      return [r, g, b];
    }
  },
  twilight: {
    name: "Twilight",
    fn: (t) => {
      if (t >= 1) return [0, 0, 0];
      return hslToRgb(260 + t * 100, 70, 10 + t * 45);
    }
  },
  monochrome: {
    name: "Cyan Mono",
    fn: (t) => {
      if (t >= 1) return [0, 0, 0];
      return [0, Math.floor(t * 220), Math.floor(t * 255)];
    }
  },
  lava: {
    name: "Lava",
    fn: (t) => {
      if (t >= 1) return [0, 0, 0];
      const r = Math.min(255, Math.floor(t * 2.5 * 255));
      const g = Math.min(255, Math.floor(Math.max(0, t * 2.0 - 0.75) * 255));
      const b = Math.min(255, Math.floor(Math.max(0, t * 3.0 - 2.5) * 255));
      return [r, g, b];
    }
  },
  neon_acid: {
    name: "Neon Acid",
    fn: (t) => {
      if (t >= 1) return [0, 0, 0];
      const r = Math.floor(Math.sin(t * Math.PI * 4 + 1.0) * 127 + 128);
      const g = Math.floor(Math.sin(t * Math.PI * 4 + 3.0) * 127 + 128);
      const b = Math.floor(Math.sin(t * Math.PI * 4 + 5.0) * 127 + 128);
      return [r, g, b];
    }
  },
  frozen: {
    name: "Frozen",
    fn: (t) => {
      if (t >= 1) return [0, 0, 0];
      return hslToRgb(200 + t * 30, 60, 40 + t * 55);
    }
  },
  vaporwave: {
    name: "Vaporwave",
    fn: (t) => {
      if (t >= 1) return [0, 0, 0];
      const r = Math.floor((0.5 + 0.5 * Math.sin(t * Math.PI * 4 + 0.0)) * 255);
      const g = Math.floor((0.2 + 0.3 * Math.sin(t * Math.PI * 4 + 2.5)) * 255);
      const b = Math.floor((0.7 + 0.3 * Math.sin(t * Math.PI * 4 + 1.0)) * 255);
      return [r, g, b];
    }
  },
  forest: {
    name: "Forest",
    fn: (t) => {
      if (t >= 1) return [0, 0, 0];
      return hslToRgb(90 + t * 50, 65, 8 + t * 45);
    }
  },
  cosine_gradient: {
    name: "Cosine Gradient",
    fn: (t) => {
      if (t >= 1) return [0, 0, 0];
      const TAU = Math.PI * 2;
      const r = Math.floor((0.5 + 0.5 * Math.cos(TAU * (t + 0.00))) * 255);
      const g = Math.floor((0.5 + 0.5 * Math.cos(TAU * (t + 0.33))) * 255);
      const b = Math.floor((0.5 + 0.5 * Math.cos(TAU * (t + 0.67))) * 255);
      return [r, g, b];
    }
  },
  inferno: {
    name: "Inferno",
    fn: (t) => {
      if (t >= 1) return [0, 0, 0];
      const r = Math.min(255, Math.floor((t * 1.5 + 0.1 * Math.sin(t * 12.0)) * 255));
      const g = Math.max(0, Math.floor((t * t * 1.2 - 0.1) * 255));
      const b = Math.max(0, Math.floor(Math.sin(t * Math.PI) * 0.8 * 255));
      return [r, g, b];
    }
  },
  gold: {
    name: "Gold",
    fn: (t) => {
      if (t >= 1) return [0, 0, 0];
      const r = Math.min(255, Math.floor(t * 2.5 * 255));
      const g = Math.min(255, Math.floor(t * t * 2.0 * 0.85 * 255));
      const b = Math.min(255, Math.floor(Math.max(0, t * t * t * 3.0 - 0.2) * 0.3 * 255));
      return [r, g, b];
    }
  },
  plasma: {
    name: "Plasma",
    fn: (t) => {
      if (t >= 1) return [0, 0, 0];
      const TAU = Math.PI * 2;
      const r = Math.floor((0.5 + 0.5 * Math.cos(TAU * (1.0 * t + 0.80))) * 255);
      const g = Math.floor((0.5 + 0.5 * Math.cos(TAU * (1.0 * t + 0.90))) * 255);
      const b = Math.floor((0.5 + 0.5 * Math.cos(TAU * (0.5 * t + 0.30))) * 255);
      return [r, g, b];
    }
  },
  zebra: {
    name: "Zebra",
    fn: (t) => {
      if (t >= 1) return [0, 0, 0];
      const bands = 20.0;
      const v = (t * bands - Math.floor(t * bands)) >= 0.5 ? 255 : 0;
      return [v, v, v];
    }
  },
  nebula: {
    name: "Nebula",
    fn: (t) => {
      if (t >= 1) return [0, 0, 0];
      const TAU = Math.PI * 2;
      const r = Math.floor((0.5 + 0.5 * Math.cos(TAU * (t * 0.8 + 0.5))) * 255);
      const g = Math.floor((0.2 * t + 0.1) * 255);
      const b = Math.floor((0.5 + 0.5 * Math.cos(TAU * (t * 0.6 + 0.0))) * 255);
      return [r, g, b];
    }
  }
};

// -- Notable Julia set presets --

const juliaPresets = [
  { name: "Douady Rabbit", c: { re: -0.1226, im: 0.7449 } },
  { name: "Dendrite", c: { re: 0, im: 1 } },
  { name: "Siegel Disc", c: { re: -0.391, im: -0.587 } },
  { name: "San Marco", c: { re: -0.75, im: 0 } },
  { name: "Spiral", c: { re: 0.285, im: 0.01 } },
  { name: "Starfish", c: { re: -0.8, im: 0.156 } },
  { name: "Lightning", c: { re: -0.4, im: 0.6 } },
  { name: "Galaxy", c: { re: 0.355, im: 0.355 } },
];

// -- L-system presets --

const lsystemPresets = {
  koch: {
    name: "Koch Curve",
    axiom: "F",
    rules: { F: "F+F-F-F+F" },
    angle: 90,
    maxIter: 6,
  },
  sierpinskiTri: {
    name: "Sierpinski Triangle",
    axiom: "F-G-G",
    rules: { F: "F-G+F+G-F", G: "GG" },
    angle: 120,
    maxIter: 7,
  },
  dragon: {
    name: "Dragon Curve",
    axiom: "FX",
    rules: { X: "X+YF+", Y: "-FX-Y" },
    angle: 90,
    maxIter: 12,
  },
  hilbert: {
    name: "Hilbert Curve",
    axiom: "A",
    rules: { A: "-BF+AFA+FB-", B: "+AF-BFB-FA+" },
    angle: 90,
    maxIter: 6,
  },
  plant: {
    name: "Branching Plant",
    axiom: "X",
    rules: { X: "F+[[X]-X]-F[-FX]+X", F: "FF" },
    angle: 25,
    maxIter: 6,
  },
  penrose: {
    name: "Penrose Tiling",
    axiom: "[F]++[F]++[F]++[F]++[F]",
    rules: { F: "G++F----G[--F----G]++", G: "-F++G[+++F++G]-" },
    angle: 36,
    maxIter: 5,
  },
  tree: {
    name: "Fractal Tree",
    axiom: "F",
    rules: { F: "FF+[+F-F-F]-[-F+F+F]" },
    angle: 22,
    maxIter: 5,
  },
  kochSnowflake: {
    name: "Koch Snowflake",
    axiom: "F--F--F",
    rules: { F: "F+F--F+F" },
    angle: 60,
    maxIter: 5,
  },
  sierpinskiArrow: {
    name: "Sierpinski Arrowhead",
    axiom: "A",
    rules: { A: "B-A-B", B: "A+B+A" },
    angle: 60,
    maxIter: 8,
  },
  gosper: {
    name: "Gosper Curve",
    axiom: "A",
    rules: { A: "A-B--B+A++AA+B-", B: "+A-BB--B-A++A+B" },
    angle: 60,
    maxIter: 4,
  },
  peano: {
    name: "Peano Curve",
    axiom: "X",
    rules: { X: "XFYFX+F+YFXFY-F-XFYFX", Y: "YFXFY-F-XFYFX+F+YFXFY" },
    angle: 90,
    maxIter: 4,
  },
  moore: {
    name: "Moore Curve",
    axiom: "LFL+F+LFL",
    rules: { L: "-RF+LFL+FR-", R: "+LF-RFR-FL+" },
    angle: 90,
    maxIter: 4,
  },
  minkowski: {
    name: "Minkowski Sausage",
    axiom: "F",
    rules: { F: "F+F-F-FF+F+F-F" },
    angle: 90,
    maxIter: 4,
  },
};

// -- Notable Mandelbrot locations --

const notableLocations = [
  { name: "Full Set", viewport: { left: -2.5, top: -1.25, width: 3.5, height: 2.5 } },
  { name: "Seahorse Valley", viewport: { left: -0.775, top: -0.135, width: 0.05, height: 0.035 } },
  { name: "Elephant Valley", viewport: { left: 0.275, top: -0.01, width: 0.015, height: 0.0105 } },
  { name: "Double Spiral", viewport: { left: -0.0452, top: -0.9868, width: 0.005, height: 0.0035 } },
  { name: "Lightning", viewport: { left: -1.786, top: -0.00005, width: 0.0001, height: 0.00007 } },
  { name: "Satellite Julia", viewport: { left: -0.163, top: -1.0378, width: 0.004, height: 0.0028 } },
  { name: "Period-3 Bulb", viewport: { left: -0.135, top: -0.665, width: 0.08, height: 0.056 } },
  { name: "Needle Tip", viewport: { left: -1.788, top: -0.0001, width: 0.0005, height: 0.00035 } },
  { name: "Mini Mandelbrot", viewport: { left: -1.7685, top: -0.0018, width: 0.004, height: 0.0028 } },
  { name: "Antenna Detail", viewport: { left: -1.255, top: -0.0225, width: 0.03, height: 0.021 } },
];

// -- L-system engine --

const lsystemRewrite = (str, rules) => {
  let result = "";
  for (let i = 0; i < str.length; i++) {
    const ch = str[i];
    result += rules[ch] !== undefined ? rules[ch] : ch;
  }
  return result;
};

const lsystemGenerate = (axiom, rules, iterations) => {
  return iterate((s) => lsystemRewrite(s, rules), iterations, axiom);
};

// Characters that mean "draw forward"
const isDrawChar = (ch) => ch === "F" || ch === "G" || ch === "A" || ch === "B";

const lsystemInterpret = (str, angleDeg, stepLength) => {
  const angleRad = (angleDeg * Math.PI) / 180;
  const commands = [];
  let sx = 0, sy = 0, sa = -Math.PI / 2;
  const stack = [];
  let minX = 0, maxX = 0, minY = 0, maxY = 0;

  for (let i = 0; i < str.length; i++) {
    const ch = str[i];
    if (isDrawChar(ch)) {
      const nx = sx + Math.cos(sa) * stepLength;
      const ny = sy + Math.sin(sa) * stepLength;
      commands.push({ type: "line", x1: sx, y1: sy, x2: nx, y2: ny });
      sx = nx; sy = ny;
      if (nx < minX) minX = nx; if (nx > maxX) maxX = nx;
      if (ny < minY) minY = ny; if (ny > maxY) maxY = ny;
    } else if (ch === "f") {
      const nx = sx + Math.cos(sa) * stepLength;
      const ny = sy + Math.sin(sa) * stepLength;
      commands.push({ type: "move", x: nx, y: ny });
      sx = nx; sy = ny;
      if (nx < minX) minX = nx; if (nx > maxX) maxX = nx;
      if (ny < minY) minY = ny; if (ny > maxY) maxY = ny;
    } else if (ch === "+") {
      sa += angleRad;
    } else if (ch === "-") {
      sa -= angleRad;
    } else if (ch === "[") {
      stack.push({ x: sx, y: sy, a: sa });
    } else if (ch === "]") {
      if (stack.length > 0) {
        const s = stack.pop();
        sx = s.x; sy = s.y; sa = s.a;
        commands.push({ type: "move", x: sx, y: sy });
      }
    }
  }
  return { commands, bounds: { minX, maxX, minY, maxY } };
};

// -- Recursive fractal drawing functions --

const drawSierpinski = (ctx, depth, maxDepth, ax, ay, bx, by, cx, cy, color) => {
  if (depth >= maxDepth) {
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.lineTo(cx, cy);
    ctx.closePath();
    const t = depth / Math.max(maxDepth, 1);
    const [r, g, b] = color(t);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fill();
    return;
  }
  const abx = (ax + bx) / 2, aby = (ay + by) / 2;
  const bcx = (bx + cx) / 2, bcy = (by + cy) / 2;
  const acx = (ax + cx) / 2, acy = (ay + cy) / 2;
  drawSierpinski(ctx, depth + 1, maxDepth, ax, ay, abx, aby, acx, acy, color);
  drawSierpinski(ctx, depth + 1, maxDepth, abx, aby, bx, by, bcx, bcy, color);
  drawSierpinski(ctx, depth + 1, maxDepth, acx, acy, bcx, bcy, cx, cy, color);
};

const kochPoints = (depth, x1, y1, x2, y2) => {
  if (depth === 0) return [{ x: x1, y: y1 }, { x: x2, y: y2 }];
  const dx = x2 - x1, dy = y2 - y1;
  const ax = x1 + dx / 3, ay = y1 + dy / 3;
  const bx = x1 + (2 * dx) / 3, by = y1 + (2 * dy) / 3;
  const px = (ax + bx) / 2 - (Math.sqrt(3) / 6) * dy;
  const py = (ay + by) / 2 + (Math.sqrt(3) / 6) * dx;
  const s1 = kochPoints(depth - 1, x1, y1, ax, ay);
  const s2 = kochPoints(depth - 1, ax, ay, px, py);
  const s3 = kochPoints(depth - 1, px, py, bx, by);
  const s4 = kochPoints(depth - 1, bx, by, x2, y2);
  return [...s1.slice(0, -1), ...s2.slice(0, -1), ...s3.slice(0, -1), ...s4];
};

const drawKochSnowflake = (ctx, depth, cx, cy, size, color) => {
  const h = size * Math.sqrt(3) / 2;
  const vertices = [
    { x: cx, y: cy - h * 2 / 3 },
    { x: cx + size / 2, y: cy + h / 3 },
    { x: cx - size / 2, y: cy + h / 3 },
  ];
  const edges = [
    kochPoints(depth, vertices[0].x, vertices[0].y, vertices[1].x, vertices[1].y),
    kochPoints(depth, vertices[1].x, vertices[1].y, vertices[2].x, vertices[2].y),
    kochPoints(depth, vertices[2].x, vertices[2].y, vertices[0].x, vertices[0].y),
  ];
  const allPoints = [...edges[0].slice(0, -1), ...edges[1].slice(0, -1), ...edges[2].slice(0, -1)];
  ctx.beginPath();
  ctx.moveTo(allPoints[0].x, allPoints[0].y);
  for (let i = 1; i < allPoints.length; i++) {
    ctx.lineTo(allPoints[i].x, allPoints[i].y);
  }
  ctx.closePath();
  const t = Math.min(1, depth / 6);
  const [r, g, b] = color(t);
  ctx.strokeStyle = `rgb(${r},${g},${b})`;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.fillStyle = `rgba(${r},${g},${b},0.08)`;
  ctx.fill();
};

const drawSierpinskiCarpet = (ctx, depth, maxDepth, x, y, size, color) => {
  if (depth >= maxDepth) {
    const t = depth / Math.max(maxDepth, 1);
    const [r, g, b] = color(t);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(x, y, size, size);
    return;
  }
  const s = size / 3;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      if (i === 1 && j === 1) continue;
      drawSierpinskiCarpet(ctx, depth + 1, maxDepth, x + i * s, y + j * s, s, color);
    }
  }
};

const drawCantorSet = (ctx, depth, maxDepth, x, y, length, color) => {
  const t = depth / Math.max(maxDepth, 1);
  const [r, g, b] = color(t);
  ctx.fillStyle = `rgb(${r},${g},${b})`;
  ctx.fillRect(x, y, length, 10);
  if (depth >= maxDepth) return;
  const newLen = length / 3;
  drawCantorSet(ctx, depth + 1, maxDepth, x, y + 30, newLen, color);
  drawCantorSet(ctx, depth + 1, maxDepth, x + 2 * newLen, y + 30, newLen, color);
};

const drawHTree = (ctx, depth, maxDepth, x, y, length, isHorizontal, color) => {
  if (depth >= maxDepth) return;
  const t = depth / Math.max(maxDepth, 1);
  const [r, g, b] = color(t);
  ctx.strokeStyle = `rgb(${r},${g},${b})`;
  ctx.lineWidth = Math.max(1, maxDepth - depth);
  ctx.beginPath();
  const half = length / 2;
  let x1, y1, x2, y2;
  if (isHorizontal) {
    x1 = x - half; x2 = x + half; y1 = y; y2 = y;
  } else {
    x1 = x; x2 = x; y1 = y - half; y2 = y + half;
  }
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  const nextLen = length / Math.sqrt(2);
  drawHTree(ctx, depth + 1, maxDepth, x1, y1, nextLen, !isHorizontal, color);
  drawHTree(ctx, depth + 1, maxDepth, x2, y2, nextLen, !isHorizontal, color);
};

const drawBarnsleyFern = (ctx, iterations, w, h, params, color) => {
  let x = 0, y = 0;
  const [r, g, b] = color(0.5);
  ctx.fillStyle = `rgb(${r},${g},${b})`;

  const totalPoints = iterations * params.density * 1000;

  for (let i = 0; i < totalPoints; i++) {
    const rnd = Math.random();
    let nx, ny;

    if (rnd < 0.01) {
      // Stem
      nx = 0;
      ny = 0.16 * y;
    } else if (rnd < 0.86) {
      // Successive leaves: params modify scale (0.85) and rotation (0.04)
      nx = params.scale * x + params.bend * y;
      ny = -params.bend * x + params.scale * y + 1.6;
    } else if (rnd < 0.93) {
      // Left leaflet
      nx = 0.2 * x - 0.26 * y;
      ny = 0.23 * x + 0.22 * y + 1.6;
    } else {
      // Right leaflet
      nx = -0.15 * x + 0.28 * y;
      ny = 0.26 * x + 0.24 * y + 0.44;
    }

    x = nx; y = ny;
    const px = w / 2 + x * (w / 11);
    const py = h - 20 - y * (h / 11);
    ctx.fillRect(px, py, 1, 1);
  }
};

// Dragon curve: iterative construction to avoid stack overflow at high depths
const drawDragonCurve = (ctx, depth, startX, startY, endX, endY, color) => {
  let pts = [{ x: startX, y: startY }, { x: endX, y: endY }];
  let signs = [1];

  for (let d = 0; d < depth; d++) {
    const np = [];
    const ns = [];
    for (let i = 0; i < pts.length - 1; i++) {
      const p1 = pts[i], p2 = pts[i + 1];
      const s = signs[i];
      const mx = (p1.x + p2.x) / 2 + s * (p2.y - p1.y) / 2;
      const my = (p1.y + p2.y) / 2 - s * (p2.x - p1.x) / 2;
      np.push(p1, { x: mx, y: my });
      ns.push(1, -1);
    }
    np.push(pts[pts.length - 1]);
    pts = np;
    signs = ns;
    if (pts.length > 600000) break;
  }

  const totalSegs = pts.length - 1;
  for (let i = 0; i < totalSegs; i++) {
    const t = totalSegs > 0 ? i / totalSegs : 0;
    const [r, g, b] = color(t);
    ctx.strokeStyle = `rgb(${r},${g},${b})`;
    ctx.beginPath();
    ctx.moveTo(pts[i].x, pts[i].y);
    ctx.lineTo(pts[i + 1].x, pts[i + 1].y);
    ctx.stroke();
  }
};

const drawFractalTree = (ctx, depth, maxDepth, x, y, len, angle, params, color) => {
  if (depth <= 0 || len < 1) return;
  const rad = (angle * Math.PI) / 180;
  const ex = x + Math.cos(rad) * len;
  const ey = y - Math.sin(rad) * len;
  const t = 1 - depth / maxDepth;
  const [r, g, b] = color(t);
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(ex, ey);
  ctx.strokeStyle = `rgb(${r},${g},${b})`;
  ctx.lineWidth = Math.max(1, depth * 0.8);
  ctx.stroke();
  const newLen = len * params.ratio;
  const randAngle = params.randomness > 0 ? (Math.random() - 0.5) * 2 * params.randomness : 0;
  const randLen = params.randomness > 0 ? 1 + (Math.random() - 0.5) * params.randomness * 0.3 : 1;
  drawFractalTree(ctx, depth - 1, maxDepth, ex, ey, newLen * randLen, angle + params.angle + randAngle, params, color);
  drawFractalTree(ctx, depth - 1, maxDepth, ex, ey, newLen * randLen, angle - params.angle + randAngle, params, color);
};

// -- Code snippets for the code panel --

const codeSnippets = {
  complex: `// Complex number arithmetic -- pure functions
const cAdd = (a, b) => ({
  re: a.re + b.re,
  im: a.im + b.im
});
const cMul = (a, b) => ({
  re: a.re * b.re - a.im * b.im,
  im: a.re * b.im + a.im * b.re
});
const cMagSq = (z) => z.re * z.re + z.im * z.im;`,

  iterate: `// Higher-order iterate: apply f to initial value n times
// This generalises Mandelbrot, Julia, and L-system iteration.
const iterate = (f, n, initial) => {
  let result = initial;
  for (let i = 0; i < n; i++) result = f(result);
  return result;
};`,

  mandelbrot: `// Mandelbrot escape count with smooth colouring
// Pure function: (c, maxIter, bailout) => { escaped, count }
const mandelbrotEscape = (c, maxIter, bailout) => {
  let z = { re: 0, im: 0 };
  for (let i = 0; i < maxIter; i++) {
    z = cAdd(cMul(z, z), c);    // z = z^2 + c
    if (cMagSq(z) > bailout * bailout) {
      const mag = Math.sqrt(cMagSq(z));
      // Fractional escape for smooth gradients
      const smooth = i + 1 - Math.log(Math.log(mag)) / Math.log(2);
      return { escaped: true, count: smooth };
    }
  }
  return { escaped: false, count: maxIter };
};`,

  julia: `// Julia set iteration -- same as Mandelbrot
// but c is fixed and z starts at the pixel coordinate.
const juliaEscape = (z0, c, maxIter, bailout) => {
  let z = { re: z0.re, im: z0.im };
  for (let i = 0; i < maxIter; i++) {
    z = cAdd(cMul(z, z), c);
    if (cMagSq(z) > bailout * bailout) {
      const smooth = i + 1 - Math.log(Math.log(cMag(z))) / Math.log(2);
      return { escaped: true, count: smooth };
    }
  }
  return { escaped: false, count: maxIter };
};`,

  palette: `// Colour palette -- pure function: (t) => [r, g, b]
// t is normalised iteration count in [0, 1]
const palette = (t) => {
  if (t >= 1) return [0, 0, 0]; // interior = black
  return hslToRgb(220 + t * 360 * 3, 85, 15 + t * 55);
};`,

  sierpinski: `// Sierpinski triangle -- recursive subdivision
// Base case: depth reached, draw filled triangle
// Recursive case: subdivide into 3 at midpoints
const drawSierpinski = (ctx, depth, maxDepth, ax, ay, bx, by, cx, cy) => {
  if (depth >= maxDepth) {           // <-- base case
    ctx.beginPath();
    ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.lineTo(cx, cy);
    ctx.closePath(); ctx.fill();
    return;
  }
  const abx = (ax+bx)/2, aby = (ay+by)/2;  // midpoints
  const bcx = (bx+cx)/2, bcy = (by+cy)/2;
  const acx = (ax+cx)/2, acy = (ay+cy)/2;
  // Three recursive calls -- one per surviving sub-triangle
  drawSierpinski(ctx, depth+1, maxDepth, ax, ay, abx, aby, acx, acy);
  drawSierpinski(ctx, depth+1, maxDepth, abx, aby, bx, by, bcx, bcy);
  drawSierpinski(ctx, depth+1, maxDepth, acx, acy, bcx, bcy, cx, cy);
};`,

  koch: `// Koch snowflake -- recursive edge replacement
// Base case: depth 0, return endpoints
// Recursive case: split edge into 4 sub-edges with a peak
const kochPoints = (depth, x1, y1, x2, y2) => {
  if (depth === 0) return [{x:x1,y:y1}, {x:x2,y:y2}];  // base
  const dx = x2-x1, dy = y2-y1;
  const ax = x1+dx/3, ay = y1+dy/3;         // 1/3 point
  const bx = x1+2*dx/3, by = y1+2*dy/3;     // 2/3 point
  const px = (ax+bx)/2 - Math.sqrt(3)/6*dy;  // peak
  const py = (ay+by)/2 + Math.sqrt(3)/6*dx;
  return [                                    // 4 recursive calls
    ...kochPoints(depth-1, x1, y1, ax, ay),
    ...kochPoints(depth-1, ax, ay, px, py),
    ...kochPoints(depth-1, px, py, bx, by),
    ...kochPoints(depth-1, bx, by, x2, y2),
  ];
};`,

  dragon: `// Dragon curve -- iterative midpoint displacement
// Each iteration doubles segment count by folding at midpoints.
// Avoids deep recursion by building the point array iteratively.
const buildDragon = (depth, x1, y1, x2, y2) => {
  let pts = [{x:x1,y:y1}, {x:x2,y:y2}];
  let signs = [1];
  for (let d = 0; d < depth; d++) {
    const np = [], ns = [];
    for (let i = 0; i < pts.length-1; i++) {
      const [p1,p2] = [pts[i], pts[i+1]];
      const s = signs[i];
      const mx = (p1.x+p2.x)/2 + s*(p2.y-p1.y)/2;
      const my = (p1.y+p2.y)/2 - s*(p2.x-p1.x)/2;
      np.push(p1, {x:mx, y:my});
      ns.push(1, -1);
    }
    np.push(pts[pts.length-1]);
    pts = np; signs = ns;
  }
  return pts; // 2^depth + 1 points
};`,

  tree: `// Fractal tree -- recursive branching
// Base case: depth 0 or branch too short
// Recursive case: draw trunk, branch left and right
const drawTree = (ctx, depth, x, y, len, angle, params) => {
  if (depth <= 0 || len < 1) return;              // <-- base case
  const rad = angle * Math.PI / 180;
  const ex = x + Math.cos(rad) * len;
  const ey = y - Math.sin(rad) * len;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(ex, ey);
  ctx.stroke();
  const newLen = len * params.ratio;
  drawTree(ctx, depth-1, ex, ey, newLen, angle+params.angle, params);
  drawTree(ctx, depth-1, ex, ey, newLen, angle-params.angle, params);
};`,

  lsystemRewrite: `// L-system string rewriting
// Pure function: (string, rules) => string
const rewrite = (str, rules) => {
  let result = "";
  for (let i = 0; i < str.length; i++) {
    const ch = str[i];
    result += rules[ch] !== undefined ? rules[ch] : ch;
  }
  return result;
};

// Apply rewriting n times using the higher-order iterate:
const generate = (axiom, rules, n) =>
  iterate(s => rewrite(s, rules), n, axiom);`,

  lsystemInterpret: `// Turtle graphics interpretation -- a fold over the string
// State: { x, y, angle, stack }
// F/G/A/B = draw forward, + = turn left, - = turn right
// [ = push state, ] = pop state
const interpret = (str, angleDeg, stepLength) => {
  const angleRad = angleDeg * Math.PI / 180;
  let state = { x: 0, y: 0, angle: -Math.PI / 2 };
  const stack = [];
  const lines = [];
  for (const ch of str) {
    if ("FGAB".includes(ch)) {
      const nx = state.x + Math.cos(state.angle) * stepLength;
      const ny = state.y + Math.sin(state.angle) * stepLength;
      lines.push({ x1: state.x, y1: state.y, x2: nx, y2: ny });
      state = { ...state, x: nx, y: ny };
    } else if (ch === "+") state.angle += angleRad;
    else if (ch === "-") state.angle -= angleRad;
    else if (ch === "[") stack.push({ ...state });
    else if (ch === "]") state = stack.pop();
  }
  return lines;
};`,

  rendering: `// Rendering pipeline -- composition of pure functions:
// 1. Map pixel to complex coordinate (pure)
// 2. Compute escape count (pure)
// 3. Map count to colour (pure)
// 4. Write pixel to ImageData
//
// Conceptually:
//   pixels.map(([px, py]) => {
//     const c = pixelToComplex(px, py, viewport);
//     const { count } = mandelbrotEscape(c, maxIter, bailout);
//     return palette(count / maxIter);
//   });`,

  glsl: `// GPU fragment shader (GLSL) -- the same pure function,
// but executed in parallel across every pixel simultaneously.
// The GPU runs this for each fragment (pixel) independently.
//
// uniform vec4 u_viewport;  // viewport bounds
// uniform int u_maxIter;    // iteration limit
//
// void main() {
//   vec2 uv = gl_FragCoord.xy / u_resolution;
//   float cx = u_viewport.x + uv.x * u_viewport.z;
//   float cy = u_viewport.y + (1.0 - uv.y) * u_viewport.w;
//   float zr = 0.0, zi = 0.0;
//   for (int i = 0; i < maxIter; i++) {
//     float zr2 = zr*zr, zi2 = zi*zi;
//     if (zr2 + zi2 > bailout*bailout) { /* escaped */ }
//     zi = 2.0*zr*zi + ci;
//     zr = zr2 - zi2 + cr;
//   }
//   // smooth colouring, palette lookup, output
//   gl_FragColor = vec4(colour, 1.0);
// }
//
// The fragment shader is the same escape-time algorithm as
// the JavaScript version, but the GPU runs thousands of
// instances in parallel -- one per pixel. This is why WebGL
// renders the full set in milliseconds instead of seconds.`,
};

// -- Styles --
const FONT_BODY = "'Fira Code', 'JetBrains Mono', 'SF Mono', monospace";
const FONT_UI = "'DM Sans', 'Segoe UI', system-ui, sans-serif";

const COLORS = {
  bg: "#06060b",
  panel: "#0c0c14",
  panelBorder: "#1a1a2e",
  text: "#bfc2d0",
  textDim: "#6b6e80",
  accent: "#2dd4a8",
  accentDim: "#1a7a60",
  codeBackground: "#0a0a12",
  inputBg: "#10101c",
  inputBorder: "#22223a",
  canvasBg: "#000000",
  white: "#e8e8f0",
};

// -- Tooltip component --

const tipStyle = {
  display: "inline-flex", alignItems: "center", justifyContent: "center",
  width: "14px", height: "14px", borderRadius: "50%",
  border: `1px solid ${COLORS.inputBorder}`, color: COLORS.textDim,
  fontSize: "9px", fontFamily: FONT_UI, fontWeight: 600, fontStyle: "italic",
  cursor: "help", flexShrink: 0, marginLeft: "4px", position: "relative",
  lineHeight: 1, userSelect: "none",
};

const Tip = ({ text }) => {
  const [show, setShow] = useState(false);
  return (
    <span
      style={tipStyle}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      i
      {show && (
        <span style={{
          position: "absolute", bottom: "calc(100% + 6px)", left: "50%",
          transform: "translateX(-50%)", background: "#1a1a2e",
          color: COLORS.text, fontFamily: FONT_UI, fontSize: "11px",
          fontStyle: "normal", fontWeight: 400, padding: "6px 10px",
          borderRadius: "4px", whiteSpace: "normal", width: "200px",
          lineHeight: "1.45", zIndex: 100, pointerEvents: "none",
          border: `1px solid ${COLORS.inputBorder}`,
          boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
          textTransform: "none", letterSpacing: "0",
        }}>
          {text}
        </span>
      )}
    </span>
  );
};

// -- Sub-components --

const TabButton = ({ label, active, onClick, tip }) => (
  <button onClick={onClick} title={tip || undefined} style={{
    fontFamily: FONT_UI, fontSize: "12px", fontWeight: active ? 600 : 400,
    padding: "6px 14px", background: active ? COLORS.accent : "transparent",
    color: active ? COLORS.bg : COLORS.textDim,
    border: active ? "none" : `1px solid ${COLORS.panelBorder}`,
    borderRadius: "3px", cursor: "pointer", letterSpacing: "0.5px",
    textTransform: "uppercase", transition: "all 0.15s",
  }}>{label}</button>
);

const Slider = ({ label, value, min, max, step, onChange, displayValue, tip }) => (
  <div style={{ marginBottom: "10px" }}>
    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "3px", alignItems: "center" }}>
      <span style={{ display: "flex", alignItems: "center", fontFamily: FONT_UI, fontSize: "11px", color: COLORS.textDim, textTransform: "uppercase", letterSpacing: "0.5px" }}>{label}{tip && <Tip text={tip} />}</span>
      <span style={{ fontFamily: FONT_BODY, fontSize: "11px", color: COLORS.accent }}>{displayValue || value}</span>
    </div>
    <input type="range" min={min} max={max} step={step || 1} value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      style={{ width: "100%", accentColor: COLORS.accent, height: "3px" }} />
  </div>
);

const SelectInput = ({ label, value, options, onChange, tip }) => (
  <div style={{ marginBottom: "10px" }}>
    <div style={{ display: "flex", alignItems: "center", fontFamily: FONT_UI, fontSize: "11px", color: COLORS.textDim, textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "3px" }}>{label}{tip && <Tip text={tip} />}</div>
    <select value={value} onChange={(e) => onChange(e.target.value)} style={{
      width: "100%", background: COLORS.inputBg, border: `1px solid ${COLORS.inputBorder}`,
      color: COLORS.text, fontFamily: FONT_UI, fontSize: "12px", padding: "5px 8px",
      borderRadius: "3px", outline: "none",
    }}>
      {options.map((opt) => (<option key={opt.value} value={opt.value}>{opt.label}</option>))}
    </select>
  </div>
);

const SmallButton = ({ label, onClick, active, tip, style: extraStyle }) => (
  <span style={{ position: "relative", display: "inline-flex", alignItems: "center" }}>
    <button onClick={onClick} title={tip || undefined} style={{
      fontFamily: FONT_UI, fontSize: "11px", padding: "4px 10px",
      background: active ? COLORS.accent : COLORS.inputBg,
      color: active ? COLORS.bg : COLORS.text,
      border: `1px solid ${active ? COLORS.accent : COLORS.inputBorder}`,
      borderRadius: "3px", cursor: "pointer", ...extraStyle,
    }}>{label}</button>
  </span>
);

const PaletteBar = ({ paletteId }) => {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext("2d");
    const fn = palettes[paletteId]?.fn || palettes.classic.fn;
    for (let x = 0; x < c.width; x++) {
      const [r, g, b] = fn(x / c.width);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(x, 0, 1, c.height);
    }
  }, [paletteId]);
  return <canvas ref={ref} width={200} height={14} style={{ width: "100%", height: "14px", borderRadius: "2px", marginTop: "4px" }} />;
};

const CodePanel = ({ visible, snippetKeys, onClose }) => {
  if (!visible) return null;
  return (
    <div style={{
      position: "absolute", right: 0, top: 0, bottom: 0, width: "420px",
      background: COLORS.codeBackground, borderLeft: `1px solid ${COLORS.panelBorder}`,
      overflowY: "auto", zIndex: 20, padding: "16px",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
        <span style={{ fontFamily: FONT_UI, fontSize: "13px", color: COLORS.white, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.5px" }}>Source Code</span>
        <button onClick={onClose} style={{ background: "none", border: "none", color: COLORS.textDim, cursor: "pointer", fontSize: "16px", fontFamily: FONT_BODY }}>x</button>
      </div>
      {snippetKeys.map((key) => (
        <div key={key} style={{ marginBottom: "16px" }}>
          <pre style={{
            fontFamily: FONT_BODY, fontSize: "11px", lineHeight: "1.6", color: COLORS.text,
            background: COLORS.panel, padding: "12px", borderRadius: "3px",
            border: `1px solid ${COLORS.panelBorder}`, whiteSpace: "pre-wrap",
            overflowX: "auto", margin: 0,
          }}>{codeSnippets[key] || ""}</pre>
        </div>
      ))}
    </div>
  );
};

const CoordDisplay = ({ coord, renderTime }) => (
  <div style={{
    position: "absolute", bottom: "8px", left: "8px", fontFamily: FONT_BODY, fontSize: "11px",
    color: COLORS.accent, background: "rgba(6,6,11,0.85)", padding: "4px 8px",
    borderRadius: "2px", pointerEvents: "none", zIndex: 10,
  }}>
    {coord && (<span>{coord.re >= 0 ? " " : ""}{coord.re.toFixed(10)} {coord.im >= 0 ? "+" : "-"} {Math.abs(coord.im).toFixed(10)}i</span>)}
    {renderTime != null && (<span style={{ marginLeft: "12px", color: COLORS.textDim }}>{renderTime}ms</span>)}
  </div>
);

const ViewportDisplay = ({ viewport }) => (
  <div style={{
    position: "absolute", top: "8px", left: "8px", fontFamily: FONT_BODY, fontSize: "10px",
    color: COLORS.textDim, background: "rgba(6,6,11,0.85)", padding: "4px 8px",
    borderRadius: "2px", pointerEvents: "none", zIndex: 10,
  }}>
    [{viewport.left.toFixed(6)}, {viewport.top.toFixed(6)}] w:{viewport.width.toExponential(3)}
  </div>
);

const MandelbrotThumbnail = ({ width, height, c, onSelect }) => {
  const ref = useRef(null);
  const vp = useMemo(() => ({ left: -2.2, top: -1.2, width: 3.4, height: 2.4 }), []);
  useEffect(() => {
    const canvas = ref.current; if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    const imgData = ctx.createImageData(w, h);
    const d = imgData.data;
    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const pt = pixelToComplex(px, py, vp, w, h);
        const result = mandelbrotEscape(pt, 80, 2);
        const idx = (py * w + px) * 4;
        if (!result.escaped) { d[idx] = 15; d[idx + 1] = 15; d[idx + 2] = 20; d[idx + 3] = 255; }
        else { const t = result.count / 80; const [r, g, b] = palettes.classic.fn(t); d[idx] = r; d[idx + 1] = g; d[idx + 2] = b; d[idx + 3] = 255; }
      }
    }
    ctx.putImageData(imgData, 0, 0);
    if (c) {
      const px = complexToPixel(c, vp, w, h);
      ctx.fillStyle = COLORS.accent;
      ctx.beginPath(); ctx.arc(px.x, px.y, 3, 0, Math.PI * 2); ctx.fill();
    }
  }, [c, vp]);
  const handleClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * width;
    const py = ((e.clientY - rect.top) / rect.height) * height;
    onSelect(pixelToComplex(px, py, vp, width, height));
  };
  return <canvas ref={ref} width={width} height={height} onClick={handleClick}
    style={{ width: "100%", height: "auto", cursor: "crosshair", borderRadius: "3px", border: `1px solid ${COLORS.panelBorder}` }} />;
};

// -- Main App --

const DEFAULT_MB_VP = { left: -2.5, top: -1.25, width: 3.5, height: 2.5 };
const DEFAULT_JL_VP = { left: -2, top: -1.5, width: 4, height: 3 };

export default function App() {
  const [mode, setMode] = useState("mandelbrot");
  const [showCode, setShowCode] = useState(false);

  const [mbViewport, setMbViewport] = useState({ ...DEFAULT_MB_VP });
  // 
  const [mbEquation, setMbEquation] = useState(0);
  // 
  const [mbMaxIter, setMbMaxIter] = useState(200);
  const [mbBailout, setMbBailout] = useState(2);
  const [mbPalette, setMbPalette] = useState("classic");
  const [mbResolution, setMbResolution] = useState(1);

  const [juliaC, setJuliaC] = useState({ re: -0.1226, im: 0.7449 });
  const [juliaViewport, setJuliaViewport] = useState({ ...DEFAULT_JL_VP });
  const [juliaMaxIter, setJuliaMaxIter] = useState(200);
  const [juliaPalette, setJuliaPalette] = useState("classic");
  const [juliaAnimating, setJuliaAnimating] = useState(false);
  const [juliaAnimSpeed, setJuliaAnimSpeed] = useState(0.5);

  const [recType, setRecType] = useState("sierpinski");
  const [recDepth, setRecDepth] = useState(5);
  const [recPalette, setRecPalette] = useState("electric");
  const [treeAngle, setTreeAngle] = useState(30);
  const [treeRatio, setTreeRatio] = useState(0.67);
  const [treeRandom, setTreeRandom] = useState(0);
  const [recAnimating, setRecAnimating] = useState(false);
  const [recAnimDepth, setRecAnimDepth] = useState(0);

  const [lsPreset, setLsPreset] = useState("koch");
  const [lsAxiom, setLsAxiom] = useState(lsystemPresets.koch.axiom);
  const [lsRulesStr, setLsRulesStr] = useState("F: F+F-F-F+F");
  const [lsAngle, setLsAngle] = useState(90);
  const [lsIterations, setLsIterations] = useState(3);
  const [lsPalette, setLsPalette] = useState("ocean");
  const [lsCustomMode, setLsCustomMode] = useState(false);

  const canvasRef = useRef(null);
  const glCanvasRef = useRef(null);
  const glStateRef = useRef(null);
  const containerRef = useRef(null);
  const [canvasSize, setCanvasSize] = useState({ w: 0, h: 0 });
  const [mouseCoord, setMouseCoord] = useState(null);
  const [renderTime, setRenderTime] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const dragStart = useRef(null);
  const renderIdRef = useRef(0);

  const [cpuRenderActive, setCpuRenderActive] = useState(false);

  const isEscapeMode = mode === "mandelbrot" || mode === "julia";

  const [cPower, setCPower] = useState(1);
  const [zPower, setZPower] = useState(2);

  const [fernDensity, setFernDensity] = useState(5);
  const [fernScale, setFernScale] = useState(0.85);
  const [fernBend, setFernBend] = useState(0.04);

  // Auto-clamp cPower and zPower when switching modes or equations
  useEffect(() => {
    let minC = 0.01, maxC = 10;
    if (mode === "mandelbrot" && (mbEquation === 1 || mbEquation === 2)) { minC = 0.1; maxC = 4; }
    if (cPower < minC) setCPower(minC);
    if (cPower > maxC) setCPower(maxC);
  }, [mode, mbEquation, cPower]);


  // Auto-reset CPU mode when switching fractal categories
  useEffect(() => { setCpuRenderActive(false); }, [mode]);

  // Cleanup WebGL context on unmount to prevent context exhaustion 
  // (Typical WebGL programming issue, esp with hot module replacement (HMR) tools like Vite)
  useEffect(() => {
    return () => {
      if (glStateRef.current && glStateRef.current.gl) {
        const ext = glStateRef.current.gl.getExtension("WEBGL_lose_context");
        if (ext) {
          ext.loseContext();
        }
        glStateRef.current = null;
      }
    };
  }, []);

  // Auto-clamp cPower when switching modes or equations
  useEffect(() => {
    let min = 0.01, max = 10;
    if (mode === "mandelbrot") {
      if (mbEquation === 1 || mbEquation === 2) { min = 0.1; max = 4; }
    }
    if (cPower < min) setCPower(min);
    if (cPower > max) setCPower(max);
  }, [mode, mbEquation, cPower]);


  const loadLsPreset = useCallback((key) => {
    const p = lsystemPresets[key]; if (!p) return;
    setLsPreset(key);
    setLsAxiom(p.axiom);
    setLsRulesStr(Object.entries(p.rules).map(([k, v]) => `${k}: ${v}`).join("\n"));
    setLsAngle(p.angle);
    setLsIterations(Math.min(p.maxIter, 4));
    setLsCustomMode(false);
  }, []);

  const parseLsRules = useCallback((str) => {
    const rules = {};
    str.split("\n").forEach((line) => {
      const idx = line.indexOf(":");
      if (idx >= 1) {
        const key = line.substring(0, idx).trim();
        const val = line.substring(idx + 1).trim();
        if (key.length >= 1) rules[key] = val;
      }
    });
    return rules;
  }, []);

  // Canvas resize with robust measurement
  useEffect(() => {
    const measure = () => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const w = Math.floor(rect.width);
      const h = Math.floor(rect.height);
      if (w > 0 && h > 0) {
        setCanvasSize((prev) => (prev.w === w && prev.h === h) ? prev : { w, h });
      }
    };
    measure();
    const timer = setTimeout(measure, 100);
    const timer2 = setTimeout(measure, 300);

    let obs;
    if (typeof ResizeObserver !== "undefined" && containerRef.current) {
      obs = new ResizeObserver(measure);
      obs.observe(containerRef.current);
    }
    window.addEventListener("resize", measure);
    return () => {
      clearTimeout(timer); clearTimeout(timer2);
      if (obs) obs.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, []);

  // WebGL escape-time rendering (GPU-accelerated)
  const renderEscapeTime = useCallback((isJulia = false) => {
    const canvas = glCanvasRef.current;
    if (!canvas || canvasSize.w === 0 || canvasSize.h === 0) return;
    canvas.width = canvasSize.w;
    canvas.height = canvasSize.h;

    let gs = glStateRef.current;
    if (!gs || gs.canvas !== canvas) {
      const gl = canvas.getContext("webgl", { antialias: false, preserveDrawingBuffer: true });
      if (!gl) {
        console.error("WebGL not supported, falling back");
        return;
      }
      const prog = buildProgram(gl);
      if (!prog) return;
      gl.useProgram(prog);
      setupQuad(gl, prog);
      gs = {
        canvas, gl, prog,
        u_resolution: gl.getUniformLocation(prog, "u_resolution"),
        u_viewport: gl.getUniformLocation(prog, "u_viewport"),
        u_maxIter: gl.getUniformLocation(prog, "u_maxIter"),
        u_bailout: gl.getUniformLocation(prog, "u_bailout"),
        u_paletteId: gl.getUniformLocation(prog, "u_paletteId"),
        u_isJulia: gl.getUniformLocation(prog, "u_isJulia"),
        u_juliaC: gl.getUniformLocation(prog, "u_juliaC"),
        u_equation: gl.getUniformLocation(prog, "u_equation"),
        u_cPower: gl.getUniformLocation(prog, "u_cPower"),
        u_zPower: gl.getUniformLocation(prog, "u_zPower"),
      };
      // Removed gl.uniform1i from here
      glStateRef.current = gs;
    }

    const { gl, prog } = gs;
    gl.useProgram(prog);
    gl.viewport(0, 0, canvas.width, canvas.height);

    const viewport = isJulia ? juliaViewport : mbViewport;
    const cvp = getCorrectedViewport(viewport, canvas.width, canvas.height);
    const maxIter = isJulia ? juliaMaxIter : mbMaxIter;
    const bailout = isJulia ? 2 : mbBailout;
    const palId = PALETTE_ID_MAP[isJulia ? juliaPalette : mbPalette] || 0;

    gl.uniform2f(gs.u_resolution, canvas.width, canvas.height);
    gl.uniform4f(gs.u_viewport, cvp.left, cvp.top, cvp.width, cvp.height);
    gl.uniform1i(gs.u_maxIter, maxIter);
    gl.uniform1f(gs.u_bailout, bailout);
    gl.uniform1i(gs.u_paletteId, palId);
    gl.uniform1i(gs.u_isJulia, isJulia ? 1 : 0);
    gl.uniform2f(gs.u_juliaC, juliaC.re, juliaC.im);

    // Dynamic assignments running every frame
    gl.uniform1i(gs.u_equation, isJulia ? 0 : mbEquation);
    gl.uniform1f(gs.u_cPower, cPower);
    gl.uniform1f(gs.u_zPower, zPower);

    const t0 = performance.now();
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.finish();
    setRenderTime(Math.round(performance.now() - t0));
  }, [canvasSize, mbViewport, mbMaxIter, mbBailout, mbPalette, mbEquation, juliaViewport, juliaMaxIter, juliaPalette, juliaC, cPower, zPower]);

  // calculate on CPU because GPU is limited to 32-bits, so renders don't work beyond a certain zoom level...
  const renderEscapeTimeCPU = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || canvasSize.w === 0 || canvasSize.h === 0) return;

    canvas.width = canvasSize.w;
    canvas.height = canvasSize.h;

    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(canvas.width, canvas.height);
    const d = imgData.data;

    const isJulia = mode === "julia";
    const viewport = isJulia ? juliaViewport : mbViewport;
    const cvp = getCorrectedViewport(viewport, canvas.width, canvas.height);
    const maxIter = isJulia ? juliaMaxIter : mbMaxIter;
    const bailout = isJulia ? 2 : mbBailout;
    const palFn = palettes[isJulia ? juliaPalette : mbPalette]?.fn || palettes.classic.fn;
    const eq = isJulia ? 0 : mbEquation;

    const t0 = performance.now();
    for (let py = 0; py < canvas.height; py++) {
      for (let px = 0; px < canvas.width; px++) {
        // const c = pixelToComplex(px, py, cvp, canvas.width, canvas.height);
        // let iter = maxIter, escaped = false, mag = 0;

        // if (isJulia) {
        //   let zr = c.re, zi = c.im;
        //   let cr = juliaC.re, ci = juliaC.im;

        //   if (cPower !== 1) {
        //     const r = Math.sqrt(cr * cr + ci * ci);
        //     if (r > 0) {
        //       const theta = Math.atan2(ci, cr);
        //       const rp = Math.pow(r, cPower);
        //       cr = rp * Math.cos(cPower * theta);
        //       ci = rp * Math.sin(cPower * theta);
        //     } else {
        //       cr = 0; ci = 0;
        //     }
        //   }

        //   const b2 = bailout * bailout;
        //   for (let i = 0; i < maxIter; i++) {
        //     const zr2 = zr * zr, zi2 = zi * zi;
        //     if (zr2 + zi2 > b2) { escaped = true; mag = Math.sqrt(zr2 + zi2); iter = i; break; }
        //     zi = 2 * zr * zi + ci;
        //     zr = zr2 - zi2 + cr;
        //   }
        // } else {
        //   let zr = 0, zi = 0;
        //   let cr = c.re, ci = c.im;

        //   if (cPower !== 1) {
        //     const r = Math.sqrt(cr * cr + ci * ci);
        //     if (r > 0) {
        //       const theta = Math.atan2(ci, cr);
        //       const rp = Math.pow(r, cPower);
        //       cr = rp * Math.cos(cPower * theta);
        //       ci = rp * Math.sin(cPower * theta);
        //     } else {
        //       cr = 0; ci = 0;
        //     }
        //   }

        //   const b2 = bailout * bailout;
        //   for (let i = 0; i < maxIter; i++) {
        //     const zr2 = zr * zr, zi2 = zi * zi;
        //     if (zr2 + zi2 > b2) { escaped = true; mag = Math.sqrt(zr2 + zi2); iter = i; break; }
        //     if (eq === 1) { zi = Math.abs(2 * zr * zi) + ci; zr = zr2 - zi2 + cr; }
        //     else if (eq === 2) { zi = -2 * zr * zi + ci; zr = zr2 - zi2 + cr; }
        //     else { zi = 2 * zr * zi + ci; zr = zr2 - zi2 + cr; }
        //   }
        // }

        // const idx = (py * canvas.width + px) * 4;
        // if (!escaped) {
        //   d[idx] = 0; d[idx + 1] = 0; d[idx + 2] = 0; d[idx + 3] = 255;
        // } else {
        //   const smooth = iter + 1 - Math.log(Math.log(mag)) / Math.log(2);
        //   const t = Math.max(0, Math.min(0.9999, smooth / maxIter));
        //   const [r, g, b] = palFn(t);
        //   d[idx] = r; d[idx + 1] = g; d[idx + 2] = b; d[idx + 3] = 255;
        // }
        // Generalize Mandelbrot to multi-brot:
        const c = pixelToComplex(px, py, cvp, canvas.width, canvas.height);
        let iter = maxIter, escaped = false, mag = 0;

        let zr = isJulia ? c.re : 0, zi = isJulia ? c.im : 0;
        let cr = isJulia ? juliaC.re : c.re, ci = isJulia ? juliaC.im : c.im;

        if (cPower !== 1) {
          const r = Math.sqrt(cr * cr + ci * ci);
          if (r > 0) {
            const theta = Math.atan2(ci, cr);
            const rp = Math.pow(r, cPower);
            cr = rp * Math.cos(cPower * theta);
            ci = rp * Math.sin(cPower * theta);
          } else { cr = 0; ci = 0; }
        }

        const b2 = bailout * bailout;
        for (let i = 0; i < maxIter; i++) {
          const r2 = zr * zr + zi * zi;
          if (r2 > b2) { escaped = true; mag = Math.sqrt(r2); iter = i; break; }

          let next_zr, next_zi;
          if (zPower !== 2) {
            let tzr = zr, tzi = zi;
            if (eq === 1) { tzr = Math.abs(zr); tzi = Math.abs(zi); }
            else if (eq === 2) { tzi = -zi; }

            const r = Math.sqrt(tzr * tzr + tzi * tzi);
            const theta = Math.atan2(tzi, tzr);
            const rp = Math.pow(r, zPower);
            next_zr = rp * Math.cos(zPower * theta);
            next_zi = rp * Math.sin(zPower * theta);
          } else {
            if (eq === 1) { next_zi = Math.abs(2 * zr * zi); next_zr = zr * zr - zi * zi; }
            else if (eq === 2) { next_zi = -2 * zr * zi; next_zr = zr * zr - zi * zi; }
            else { next_zi = 2 * zr * zi; next_zr = zr * zr - zi * zi; }
          }
          zi = next_zi + ci;
          zr = next_zr + cr;
        }

        const idx = (py * canvas.width + px) * 4;
        if (!escaped) {
          d[idx] = 0; d[idx + 1] = 0; d[idx + 2] = 0; d[idx + 3] = 255;
        } else {
          const log_p = Math.log(Math.max(1.001, zPower));
          const smooth = iter + 1 - Math.log(Math.log(mag)) / log_p;
          const t = Math.max(0, Math.min(0.9999, smooth / maxIter));
          const [r, g, b] = palFn(t);
          d[idx] = r; d[idx + 1] = g; d[idx + 2] = b; d[idx + 3] = 255;
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);
    setRenderTime(Math.round(performance.now() - t0));
  }, [canvasSize, mode, mbViewport, mbMaxIter, mbBailout, mbPalette, mbEquation, juliaViewport, juliaMaxIter, juliaPalette, juliaC, cPower, zPower]);

  useEffect(() => {
    if (cpuRenderActive && isEscapeMode) {
      const t = setTimeout(() => renderEscapeTimeCPU(), 10);
      return () => clearTimeout(t);
    }
  }, [cpuRenderActive, isEscapeMode, renderEscapeTimeCPU]);

  // Recursive fractal rendering
  const renderRecursive = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || canvas.width === 0 || canvas.height === 0) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = "#fafaf8"; ctx.fillRect(0, 0, w, h);
    const palFn = palettes[recPalette]?.fn || palettes.electric.fn;
    const depth = recAnimating ? recAnimDepth : recDepth;
    const t0 = performance.now();

    if (recType === "sierpinski") {
      const size = Math.min(w, h) * 0.85;
      const cx = w / 2, cy = h / 2, triH = size * Math.sqrt(3) / 2;
      drawSierpinski(ctx, 0, depth, cx, cy - triH * 0.6, cx + size / 2, cy + triH * 0.4, cx - size / 2, cy + triH * 0.4, palFn);
    } else if (recType === "koch") {
      drawKochSnowflake(ctx, depth, w / 2, h / 2, Math.min(w, h) * 0.6, palFn);
    } else if (recType === "dragon") {
      const size = Math.min(w, h) * 0.35;
      ctx.lineWidth = Math.max(0.5, 2 - depth * 0.1);
      drawDragonCurve(ctx, depth, w / 2 - size / 2, h / 2, w / 2 + size / 2, h / 2, palFn);
    } else if (recType === "tree") {
      drawFractalTree(ctx, depth, depth, w / 2, h * 0.88, Math.min(w, h) * 0.22, 90, { angle: treeAngle, ratio: treeRatio, randomness: treeRandom / 100 }, palFn);
    } else if (recType === "carpet") {
      const size = Math.min(w, h) * 0.8;
      drawSierpinskiCarpet(ctx, 0, depth, w / 2 - size / 2, h / 2 - size / 2, size, palFn);
    } else if (recType === "cantor") {
      const size = w * 0.8;
      drawCantorSet(ctx, 0, depth, w / 2 - size / 2, h / 2 - (depth * 30) / 2, size, palFn);
    } else if (recType === "htree") {
      const size = Math.min(w, h) * 0.5;
      drawHTree(ctx, 0, depth, w / 2, h / 2, size, true, palFn);
    } else if (recType === "fern") {
      drawBarnsleyFern(ctx, depth, w, h, {
        density: fernDensity,
        scale: fernScale,
        bend: fernBend
      }, palFn);
    }
    setRenderTime(Math.round(performance.now() - t0));
  }, [recType, recDepth, recPalette, recAnimating, recAnimDepth, treeAngle, treeRatio, treeRandom, fernDensity, fernScale, fernBend]);

  // L-system rendering
  const renderLSystem = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || canvas.width === 0 || canvas.height === 0) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = "#fafaf8"; ctx.fillRect(0, 0, w, h);
    const palFn = palettes[lsPalette]?.fn || palettes.ocean.fn;
    const t0 = performance.now();
    try {
      const rules = parseLsRules(lsRulesStr);
      const str = lsystemGenerate(lsAxiom, rules, lsIterations);
      if (str.length > 3000000) {
        ctx.fillStyle = "#666"; ctx.font = `14px ${FONT_UI}`;
        ctx.fillText("String too long (" + str.length.toLocaleString() + " chars). Reduce iterations.", 20, 40);
        setRenderTime(null); return;
      }
      const { commands, bounds } = lsystemInterpret(str, lsAngle, 10);
      const bw = bounds.maxX - bounds.minX || 1;
      const bh = bounds.maxY - bounds.minY || 1;
      const scale = Math.min((w * 0.88) / bw, (h * 0.88) / bh);
      const offX = (w - bw * scale) / 2 - bounds.minX * scale;
      const offY = (h - bh * scale) / 2 - bounds.minY * scale;
      ctx.lineWidth = Math.max(0.3, 1.5 - lsIterations * 0.2);
      const totalCmds = commands.filter(c => c.type === "line").length;
      let lineIdx = 0;
      for (const cmd of commands) {
        if (cmd.type === "line") {
          const t = totalCmds > 0 ? lineIdx / totalCmds : 0;
          const [r, g, b] = palFn(t);
          ctx.strokeStyle = `rgb(${r},${g},${b})`;
          ctx.beginPath();
          ctx.moveTo(cmd.x1 * scale + offX, cmd.y1 * scale + offY);
          ctx.lineTo(cmd.x2 * scale + offX, cmd.y2 * scale + offY);
          ctx.stroke();
          lineIdx++;
        }
      }
    } catch (e) {
      ctx.fillStyle = "#666"; ctx.font = `14px ${FONT_UI}`;
      ctx.fillText("Error: " + e.message, 20, 40);
    }
    setRenderTime(Math.round(performance.now() - t0));
  }, [lsAxiom, lsRulesStr, lsAngle, lsIterations, lsPalette, parseLsRules]);

  // Trigger rendering

  useEffect(() => {
    if (canvasSize.w === 0 || canvasSize.h === 0) return;
    if (isEscapeMode) {
      // WebGL handles its own canvas sizing inside renderEscapeTime
      if (mode === "mandelbrot") renderEscapeTime(false);
      else renderEscapeTime(true);
    } else {
      const canvas = canvasRef.current; if (!canvas) return;
      canvas.width = canvasSize.w;
      canvas.height = canvasSize.h;
      if (mode === "recursive") renderRecursive();
      else if (mode === "lsystem") renderLSystem();
    }
  }, [mode, canvasSize, renderEscapeTime, renderRecursive, renderLSystem, isEscapeMode]);

  // Julia animation
  useEffect(() => {
    if (!juliaAnimating) return;
    let raf;
    const t0 = performance.now();
    const animate = () => {
      const elapsed = (performance.now() - t0) / 1000;
      const angle = elapsed * juliaAnimSpeed * 0.5;
      setJuliaC({ re: 0.7885 * Math.cos(angle), im: 0.7885 * Math.sin(angle) });
      raf = requestAnimationFrame(animate);
    };
    raf = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(raf);
  }, [juliaAnimating, juliaAnimSpeed]);

  // Recursive animation
  useEffect(() => {
    if (!recAnimating) return;
    let depth = 0;
    const maxD = recType === "tree" ? 14 : recType === "dragon" ? 18 : recType === "koch" ? 8 : 10;
    const iv = setInterval(() => {
      depth++;
      if (depth > Math.min(recDepth, maxD)) depth = 0;
      setRecAnimDepth(depth);
    }, 700);
    return () => clearInterval(iv);
  }, [recAnimating, recDepth, recType]);

  // Pan/Zoom - deprecated, not aware of the aspect ratio of the viewport... 
  // const getVPSet = () => mode === "mandelbrot" ? setMbViewport : mode === "julia" ? setJuliaViewport : null;
  // const getVP = () => mode === "mandelbrot" ? mbViewport : mode === "julia" ? juliaViewport : null;

  // const getActiveCanvas = () => (mode === "mandelbrot" || mode === "julia") ? glCanvasRef.current : canvasRef.current;

  // const handleMouseDown = (e) => {
  //   if (mode !== "mandelbrot" && mode !== "julia") return;
  //   setIsDragging(true);
  //   dragStart.current = { x: e.clientX, y: e.clientY, viewport: { ...getVP() } };
  // };

  // const handleMouseMove = (e) => {
  //   const canvas = getActiveCanvas(); if (!canvas) return;
  //   const rect = canvas.getBoundingClientRect();
  //   const px = ((e.clientX - rect.left) / rect.width) * canvas.width;
  //   const py = ((e.clientY - rect.top) / rect.height) * canvas.height;
  //   const vp = getVP();
  //   if (vp) setMouseCoord(pixelToComplex(px, py, vp, canvas.width, canvas.height));

  //   if (isDragging && dragStart.current) {
  //     const dx = e.clientX - dragStart.current.x;
  //     const dy = e.clientY - dragStart.current.y;
  //     const svp = dragStart.current.viewport;
  //     const setVP = getVPSet();
  //     if (setVP) setVP({
  //       left: svp.left - (dx / rect.width) * svp.width,
  //       top: svp.top - (dy / rect.height) * svp.height,
  //       width: svp.width, height: svp.height,
  //     });
  //   }
  // };

  // const handleMouseUp = () => { setIsDragging(false); dragStart.current = null; };

  // const handleWheel = (e) => {
  //   if (mode !== "mandelbrot" && mode !== "julia") return;
  //   e.preventDefault();
  //   const canvas = getActiveCanvas(); if (!canvas) return;
  //   const rect = canvas.getBoundingClientRect();
  //   const px = ((e.clientX - rect.left) / rect.width) * canvas.width;
  //   const py = ((e.clientY - rect.top) / rect.height) * canvas.height;
  //   const vp = getVP(); const setVP = getVPSet();
  //   if (!vp || !setVP) return;
  //   const factor = e.deltaY > 0 ? 1.15 : 0.87;
  //   const center = pixelToComplex(px, py, vp, canvas.width, canvas.height);
  //   const nw = vp.width * factor, nh = vp.height * factor;
  //   setVP({ left: center.re - (px / canvas.width) * nw, top: center.im - (py / canvas.height) * nh, width: nw, height: nh });
  // };

  // const handleDoubleClick = (e) => {
  //   if (mode !== "mandelbrot" && mode !== "julia") return;
  //   const canvas = getActiveCanvas(); if (!canvas) return;
  //   const rect = canvas.getBoundingClientRect();
  //   const px = ((e.clientX - rect.left) / rect.width) * canvas.width;
  //   const py = ((e.clientY - rect.top) / rect.height) * canvas.height;
  //   const vp = getVP(); const setVP = getVPSet();
  //   if (!vp || !setVP) return;
  //   const center = pixelToComplex(px, py, vp, canvas.width, canvas.height);
  //   const nw = vp.width * 0.5, nh = vp.height * 0.5;
  //   setVP({ left: center.re - nw / 2, top: center.im - nh / 2, width: nw, height: nh });
  // };

  // // Zoom in/out centered on viewport center
  // const zoomView = (factor) => {
  //   const vp = getVP(); const setVP = getVPSet();
  //   if (!vp || !setVP) return;
  //   const cx = vp.left + vp.width / 2;
  //   const cy = vp.top + vp.height / 2;
  //   const nw = vp.width * factor, nh = vp.height * factor;
  //   setVP({ left: cx - nw / 2, top: cy - nh / 2, width: nw, height: nh });
  // };
  // 
  // 
  // 


  // Pan/Zoom
  // updated implementation - aspect ratio aware...
  const getVPSet = () => mode === "mandelbrot" ? setMbViewport : mode === "julia" ? setJuliaViewport : null;
  const getVP = () => mode === "mandelbrot" ? mbViewport : mode === "julia" ? juliaViewport : null;

  const handleMouseDown = (e) => {
    if (mode !== "mandelbrot" && mode !== "julia") return;
    setIsDragging(true);
    setCpuRenderActive(false);
    dragStart.current = { x: e.clientX, y: e.clientY, viewport: { ...getVP() } };
  };

  const handleMouseMove = (e) => {
    const container = containerRef.current; if (!container) return;
    const rect = container.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * canvasSize.w;
    const py = ((e.clientY - rect.top) / rect.height) * canvasSize.h;

    const vp = getVP();
    if (vp) {
      const cvp = getCorrectedViewport(vp, canvasSize.w, canvasSize.h);
      setMouseCoord(pixelToComplex(px, py, cvp, canvasSize.w, canvasSize.h));
    }

    if (isDragging && dragStart.current) {
      const dx = e.clientX - dragStart.current.x;
      const dy = e.clientY - dragStart.current.y;
      const svp = dragStart.current.viewport;
      const setVP = getVPSet();
      if (setVP) {
        const scvp = getCorrectedViewport(svp, canvasSize.w, canvasSize.h);
        setVP({
          left: svp.left - (dx / canvasSize.w) * scvp.width,
          top: svp.top - (dy / canvasSize.h) * scvp.height,
          width: svp.width, height: svp.height,
        });
      }
    }
  };

  const handleMouseUp = () => { setIsDragging(false); dragStart.current = null; };

  const handleWheel = (e) => {
    if (mode !== "mandelbrot" && mode !== "julia") return;
    e.preventDefault();
    const container = containerRef.current; if (!container) return;
    const rect = container.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * canvasSize.w;
    const py = ((e.clientY - rect.top) / rect.height) * canvasSize.h;

    const vp = getVP(); const setVP = getVPSet();
    if (!vp || !setVP) return;

    setCpuRenderActive(false);
    const cvp = getCorrectedViewport(vp, canvasSize.w, canvasSize.h);
    const center = pixelToComplex(px, py, cvp, canvasSize.w, canvasSize.h);

    const factor = e.deltaY > 0 ? 1.15 : 0.87;
    const nw = vp.width * factor;
    const nh = vp.height * factor;
    const nch = nw * (canvasSize.h / canvasSize.w);
    const new_ctop = center.im - (py / canvasSize.h) * nch;
    const new_top = new_ctop - nh / 2 + nch / 2;

    setVP({ left: center.re - (px / canvasSize.w) * nw, top: new_top, width: nw, height: nh });
  };

  const handleDoubleClick = (e) => {
    if (mode !== "mandelbrot" && mode !== "julia") return;
    const container = containerRef.current; if (!container) return;
    const rect = container.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * canvasSize.w;
    const py = ((e.clientY - rect.top) / rect.height) * canvasSize.h;

    const vp = getVP(); const setVP = getVPSet();
    if (!vp || !setVP) return;

    setCpuRenderActive(false);
    const cvp = getCorrectedViewport(vp, canvasSize.w, canvasSize.h);
    const center = pixelToComplex(px, py, cvp, canvasSize.w, canvasSize.h);
    const nw = vp.width * 0.5, nh = vp.height * 0.5;

    setVP({ left: center.re - nw / 2, top: center.im - nh / 2, width: nw, height: nh });
  };

  const zoomView = (factor) => {
    const vp = getVP(); const setVP = getVPSet();
    if (!vp || !setVP) return;
    setCpuRenderActive(false);
    const cx = vp.left + vp.width / 2;
    const cy = vp.top + vp.height / 2;
    const nw = vp.width * factor, nh = vp.height * factor;
    setVP({ left: cx - nw / 2, top: cy - nh / 2, width: nw, height: nh });
  };

  // Global Reset
  const handleGlobalReset = () => {
    // Reset Mode and Global states
    setMode("mandelbrot");
    setShowCode(false);
    setCpuRenderActive(false);
    setCPower(1);
    setZPower(2);

    // Reset Mandelbrot
    setMbViewport({ ...DEFAULT_MB_VP });
    setMbEquation(0);
    setMbMaxIter(200);
    setMbBailout(2);
    setMbPalette("classic");

    // Reset Julia
    setJuliaC({ re: -0.1226, im: 0.7449 });
    setJuliaViewport({ ...DEFAULT_JL_VP });
    setJuliaMaxIter(200);
    setJuliaPalette("classic");
    setJuliaAnimating(false);
    setJuliaAnimSpeed(0.5);

    // Reset Recursive
    setRecType("sierpinski");
    setRecDepth(5);
    setRecPalette("electric");
    setTreeAngle(30);
    setTreeRatio(0.67);
    setTreeRandom(0);
    setRecAnimating(false);
    setRecAnimDepth(0);

    // Reset L-System
    setLsPreset("koch");
    setLsAxiom(lsystemPresets.koch.axiom);
    setLsRulesStr("F: F+F-F-F+F");
    setLsAngle(90);
    setLsIterations(3);
    setLsPalette("ocean");
    setLsCustomMode(false);

    // Reset Barnsley Fern
    setFernDensity(5);
    setFernScale(0.85);
    setFernBend(0.04);
  };


  // Export
  const exportImage = (format) => {
    const canvas = (mode === "mandelbrot" || mode === "julia") ? glCanvasRef.current : canvasRef.current;
    if (!canvas) return;
    const mime = format === "jpeg" ? "image/jpeg" : "image/png";
    const ext = format === "jpeg" ? "jpg" : "png";
    const url = canvas.toDataURL(mime, format === "jpeg" ? 0.95 : undefined);
    const a = document.createElement("a");
    a.download = `shauryashaurya-corals-${mode}-${Date.now()}.${ext}`;
    a.href = url;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
  };

  // Bookmark
  const saveBookmark = () => {
    const state = { mode };
    if (mode === "mandelbrot") Object.assign(state, { vp: mbViewport, mi: mbMaxIter, pal: mbPalette });
    else if (mode === "julia") Object.assign(state, { vp: juliaViewport, mi: juliaMaxIter, pal: juliaPalette, c: juliaC });
    else if (mode === "recursive") Object.assign(state, { t: recType, d: recDepth, pal: recPalette, ta: treeAngle, tr: treeRatio, trnd: treeRandom });
    else if (mode === "lsystem") Object.assign(state, { ax: lsAxiom, ru: lsRulesStr, an: lsAngle, it: lsIterations, pal: lsPalette });
    navigator.clipboard.writeText(window.location.href.split("#")[0] + "#" + btoa(JSON.stringify(state))).catch(() => { });
  };

  const getCodeKeys = () => {
    switch (mode) {
      case "mandelbrot": return ["complex", "iterate", "mandelbrot", "palette", "rendering", "glsl"];
      case "julia": return ["complex", "iterate", "julia", "palette", "glsl"];
      case "recursive":
        if (recType === "sierpinski") return ["sierpinski"];
        if (recType === "koch") return ["koch"];
        if (recType === "dragon") return ["dragon"];
        if (recType === "tree") return ["tree"];
        return [];
      case "lsystem": return ["iterate", "lsystemRewrite", "lsystemInterpret"];
      default: return [];
    }
  };

  const paletteOptions = useMemo(() => Object.entries(palettes).map(([k, v]) => ({ value: k, label: v.name })), []);

  const renderSidebar = () => (
    <div style={{
      width: "260px", minWidth: "260px", maxWidth: "260px",
      background: COLORS.panel, borderRight: `1px solid ${COLORS.panelBorder}`,
      padding: "14px", overflowY: "auto", display: "flex", flexDirection: "column", gap: "6px", flexShrink: 0,
    }}>
      <div style={{ display: "flex", alignItems: "center", fontFamily: FONT_UI, fontSize: "11px", color: COLORS.textDim, textTransform: "uppercase", letterSpacing: "1px", marginBottom: "4px" }}>Parameters<Tip text="Controls for the current fractal mode. Adjust these values and the canvas updates in real time." /></div>

      {mode === "mandelbrot" && (<>
        <Slider label="Max Iterations" value={mbMaxIter} min={50} max={5000} step={50} onChange={setMbMaxIter}
          tip="How many times z = z^2 + c is repeated per pixel. Higher values reveal more detail at deep zoom levels but render slower." />

        <Slider label="Bailout Radius" value={mbBailout} min={2} max={20} step={0.5} onChange={setMbBailout}
          tip="The escape threshold. A point is considered escaped when |z| exceeds this value. The standard value is 2; higher values can subtly change the colouring." />

        <SelectInput label="Palette" value={mbPalette} options={paletteOptions} onChange={setMbPalette}
          tip="The colour mapping function. Maps the escape iteration count to an RGB colour. Each palette is a pure function: (count) => colour." />

        <SelectInput label="Equation Variant" value={mbEquation} options={[
          { value: 0, label: "Mandelbrot" },
          { value: 1, label: "Burning Ship" },
          { value: 2, label: "Tricorn (Mandelbar)" }
        ]} onChange={(val) => setMbEquation(Number(val))} tip="The formula used for complex iteration." />

        <Slider label="C Power" value={cPower}
          min={cBounds.min} max={cBounds.max} step={cBounds.step}
          onChange={setCPower} displayValue={cPower.toFixed(2)}
          tip="Raises the constant 'c' to a fractional power before iteration. 1 is standard." />

        <Slider label="Z Power" value={zPower} min={1.1} max={10} step={0.1} onChange={setZPower} displayValue={zPower.toFixed(1)}
          tip="Modifies the exponent of z in the iteration formula (z^p + c). 2 is the classic set, 3 is the Multibrot set." />

        <PaletteBar paletteId={mbPalette} />

        <div style={{ marginTop: "10px" }}>
          <div style={{ display: "flex", alignItems: "center", fontFamily: FONT_UI, fontSize: "11px", color: COLORS.textDim, textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "6px" }}>Notable Locations<Tip text="Pre-saved coordinates for mathematically interesting regions of the Mandelbrot set. Each one demonstrates a different structural feature: spirals, miniature copies, cusps, or branching filaments." /></div>
          <div style={{ display: "flex", flexDirection: "column", gap: "3px" }}>
            {notableLocations.map((loc) => (
              <button key={loc.name} onClick={() => setMbViewport({ ...loc.viewport })} style={{
                fontFamily: FONT_UI, fontSize: "11px", padding: "4px 8px", background: COLORS.inputBg,
                color: COLORS.text, border: `1px solid ${COLORS.inputBorder}`, borderRadius: "2px",
                cursor: "pointer", textAlign: "left",
              }}>{loc.name}</button>
            ))}
          </div>
        </div>
      </>)}

      {mode === "julia" && (<>
        <Slider label="Max Iterations" value={juliaMaxIter} min={50} max={5000} step={50} onChange={setJuliaMaxIter}
          tip="How many times z = z^2 + c is repeated per pixel. Higher values resolve finer boundary detail." />

        <Slider label="C Power" value={cPower}
          min={cBounds.min} max={cBounds.max} step={cBounds.step}
          onChange={setCPower} displayValue={cPower.toFixed(2)}
          tip="Raises the constant 'c' to a fractional power before iteration. 1 is standard." />

        <Slider label="Z Power" value={zPower} min={1.1} max={10} step={0.1} onChange={setZPower} displayValue={zPower.toFixed(1)}
          tip="Modifies the exponent of z in the iteration formula (z^p + c). 2 is the classic set, 3 is the Multibrot set." />

        <SelectInput label="Palette" value={juliaPalette} options={paletteOptions} onChange={setJuliaPalette}
          tip="The colour mapping function applied to escape counts." />

        <PaletteBar paletteId={juliaPalette} />

        <div style={{ marginTop: "8px" }}>
          <div style={{ display: "flex", alignItems: "center", fontFamily: FONT_UI, fontSize: "11px", color: COLORS.textDim, textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "4px" }}>
            c = {juliaC.re.toFixed(4)} {juliaC.im >= 0 ? "+" : "-"} {Math.abs(juliaC.im).toFixed(4)}i
            <Tip text="The fixed complex constant c in z = z^2 + c. Click on the Mandelbrot thumbnail below to choose a c value. Points near the boundary of the Mandelbrot set produce the most intricate Julia sets." />
          </div>

          <MandelbrotThumbnail width={180} height={130} c={juliaC} onSelect={setJuliaC} />
        </div>

        <div style={{ marginTop: "8px", display: "flex", gap: "4px", flexWrap: "wrap" }}>
          <SmallButton label={juliaAnimating ? "Stop" : "Animate"} onClick={() => setJuliaAnimating(!juliaAnimating)} active={juliaAnimating}
            tip="Rotate c around a circle in the complex plane, showing how the Julia set deforms continuously." />
          {juliaAnimating && <Slider label="Speed" value={juliaAnimSpeed} min={0.1} max={2} step={0.1} onChange={setJuliaAnimSpeed}
            tip="How fast c rotates around the circle." />}
        </div>

        <div style={{ marginTop: "8px" }}>
          <div style={{ display: "flex", alignItems: "center", fontFamily: FONT_UI, fontSize: "11px", color: COLORS.textDim, textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "4px" }}>Presets<Tip text="Named Julia sets at mathematically significant c values. Each has a distinctive shape reflecting its position in the Mandelbrot set." /></div>
          <div style={{ display: "flex", flexDirection: "column", gap: "3px" }}>
            {juliaPresets.map((p) => (
              <button key={p.name} onClick={() => { setJuliaC(p.c); setJuliaAnimating(false); }} style={{
                fontFamily: FONT_UI, fontSize: "11px", padding: "4px 8px", background: COLORS.inputBg,
                color: COLORS.text, border: `1px solid ${COLORS.inputBorder}`, borderRadius: "2px",
                cursor: "pointer", textAlign: "left",
              }}>{p.name}</button>
            ))}
          </div>
        </div>
      </>)}

      {mode === "recursive" && (<>
        <SelectInput label="Fractal Type" value={recType} options={[
          { value: "sierpinski", label: "Sierpinski Triangle" },
          { value: "carpet", label: "Sierpinski Carpet" },
          { value: "koch", label: "Koch Snowflake" },
          { value: "dragon", label: "Dragon Curve" },
          { value: "tree", label: "Fractal Tree" },
          { value: "htree", label: "H-Tree" },
          { value: "cantor", label: "Cantor Set" },
          { value: "fern", label: "Barnsley Fern (IFS)" },
        ]} onChange={setRecType}
          tip="Each fractal is a recursive function rendered visually. Sierpinski subdivides triangles; Koch replaces edges; Dragon folds midpoints; Tree branches." />
        <Slider label="Recursion Depth" value={recDepth} min={1}
          max={recType === "dragon" ? 18 : recType === "tree" ? 14 : recType === "koch" ? 8 : 10}
          onChange={setRecDepth}
          tip="How many levels deep the recursive function calls itself. Each increment adds one more level of detail and multiplies the number of elements." />
        <SelectInput label="Palette" value={recPalette} options={paletteOptions} onChange={setRecPalette}
          tip="Colours are mapped by recursion depth or drawing order." />
        <PaletteBar paletteId={recPalette} />
        {recType === "tree" && (<>
          <Slider label="Branch Angle" value={treeAngle} min={15} max={60} onChange={setTreeAngle} displayValue={`${treeAngle} deg`}
            tip="The angle between branches. Small angles (15-25) produce tall narrow trees; large angles (45-60) produce wide bushy trees." />
          <Slider label="Length Ratio" value={treeRatio} min={0.5} max={0.8} step={0.01} onChange={setTreeRatio} displayValue={treeRatio.toFixed(2)}
            tip="How much shorter each generation of branches is. 0.5 means each branch is half the parent. 0.8 means branches stay long, creating a denser tree." />
          <Slider label="Randomness" value={treeRandom} min={0} max={30} onChange={setTreeRandom} displayValue={`${treeRandom}%`}
            tip="Adds random variation to branch angle and length, breaking the rigid symmetry to produce natural-looking trees. Each render is unique." />
        </>)}
        {recType === "fern" && (<>
          <Slider label="Point Density" value={fernDensity} min={1} max={20} onChange={setFernDensity}
            tip="Multiplier for the number of points drawn. Higher density fills in the fractal but takes longer to render." />
          <Slider label="Frond Scale" value={fernScale} min={0.5} max={0.99} step={0.01} onChange={setFernScale} displayValue={fernScale.toFixed(2)}
            tip="The scaling of successive leaflets. Lower values make a short, stubby fern; higher values approach an infinitely long fern." />
          <Slider label="Frond Bend" value={fernBend} min={-0.2} max={0.2} step={0.01} onChange={setFernBend} displayValue={fernBend.toFixed(2)}
            tip="The rotation parameter of the main stem. Modifying this causes the entire fern to curl left or right." />
        </>)}
        <div style={{ display: "flex", gap: "4px", marginTop: "6px" }}>
          <SmallButton label={recAnimating ? "Stop" : "Animate"} onClick={() => { setRecAnimating(!recAnimating); setRecAnimDepth(0); }} active={recAnimating}
            tip="Build the fractal one recursion level at a time. Watch the recursive structure appear step by step." />
          <SmallButton label="Reset" onClick={() => setRecAnimDepth(0)} tip="Restart the animation from depth 0." />
        </div>
      </>)}

      {mode === "lsystem" && (<>
        {!lsCustomMode && <SelectInput label="Preset" value={lsPreset}
          options={Object.entries(lsystemPresets).map(([k, v]) => ({ value: k, label: v.name }))}
          onChange={loadLsPreset}
          tip="Built-in L-system definitions. Each preset defines an axiom, production rules, and turn angle that together produce a specific fractal." />}
        <Slider label="Iterations" value={lsIterations} min={1} max={8} onChange={setLsIterations}
          tip="How many times the production rules are applied to the string. Each iteration replaces every symbol, so the string grows exponentially. More iterations = finer detail but slower rendering." />
        <Slider label="Turn Angle" value={lsAngle} min={1} max={180} onChange={setLsAngle} displayValue={`${lsAngle} deg`}
          tip="The rotation angle for + and - commands in turtle graphics. This single parameter dramatically changes the shape: 90 for right angles, 60 for hexagonal, 36 for pentagonal symmetry." />
        <SelectInput label="Palette" value={lsPalette} options={paletteOptions} onChange={setLsPalette}
          tip="Colour is mapped along the drawing path, from the first line segment to the last." />
        <PaletteBar paletteId={lsPalette} />
        <div style={{ marginTop: "8px" }}>
          <SmallButton label={lsCustomMode ? "Use Presets" : "Custom Rules"} onClick={() => setLsCustomMode(!lsCustomMode)} active={lsCustomMode}
            tip="Switch to custom mode to type your own axiom and production rules. Experiment with different rules and angles." />
        </div>
        {lsCustomMode && (
          <div style={{ marginTop: "8px" }}>
            <div style={{ display: "flex", alignItems: "center", fontFamily: FONT_UI, fontSize: "11px", color: COLORS.textDim, textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "3px" }}>Axiom<Tip text="The starting string before any rules are applied. Typically a single character like F or A." /></div>
            <input value={lsAxiom} onChange={(e) => setLsAxiom(e.target.value)} style={{
              width: "100%", background: COLORS.inputBg, border: `1px solid ${COLORS.inputBorder}`,
              color: COLORS.text, fontFamily: FONT_BODY, fontSize: "12px", padding: "5px 8px",
              borderRadius: "3px", outline: "none", boxSizing: "border-box", marginBottom: "8px",
            }} />
            <div style={{ display: "flex", alignItems: "center", fontFamily: FONT_UI, fontSize: "11px", color: COLORS.textDim, textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "3px" }}>Rules<Tip text="Production rules, one per line, in the format 'X: replacement'. Each iteration replaces every occurrence of X with its replacement. Characters without rules are kept unchanged." /></div>
            <textarea value={lsRulesStr} onChange={(e) => setLsRulesStr(e.target.value)} rows={4} style={{
              width: "100%", background: COLORS.inputBg, border: `1px solid ${COLORS.inputBorder}`,
              color: COLORS.text, fontFamily: FONT_BODY, fontSize: "11px", padding: "5px 8px",
              borderRadius: "3px", outline: "none", resize: "vertical", boxSizing: "border-box",
            }} />
            <div style={{ fontFamily: FONT_UI, fontSize: "10px", color: COLORS.textDim, marginTop: "6px", lineHeight: "1.5" }}>
              F, G, A, B = draw forward | f = move | + = turn left | - = turn right | [ = save | ] = restore
            </div>
          </div>
        )}
      </>)}

      <div style={{ marginTop: "auto", paddingTop: "12px", borderTop: `1px solid ${COLORS.panelBorder}`, display: "flex", gap: "4px", flexWrap: "wrap" }}>
        {(mode === "mandelbrot" || mode === "julia") && (<>
          <SmallButton
            label="Force GPU Update"
            onClick={() => { setCpuRenderActive(false); renderEscapeTime(mode === "julia"); }}
            tip="Force the standard WebGL canvas to calculate and redraw."
          />
          <SmallButton
            label="Deep Zoom (CPU)"
            onClick={() => setCpuRenderActive(true)}
            tip="Calculate using 64-bit JS float precision. Resolves WebGL distortion at deep zoom limits. Very slow, strictly for rendering static frames."
            style={{ borderColor: COLORS.accent, color: COLORS.accent }}
          />
        </>)}
        <SmallButton label={showCode ? "Hide Code" : "Show Code"} onClick={() => setShowCode(!showCode)} active={showCode}
          tip="Show the pure functions behind the current fractal: complex arithmetic, escape counting, recursive drawing, or L-system rewriting." />
        <SmallButton label="Bookmark" onClick={saveBookmark} tip="Copy a URL with the current state (mode, viewport, parameters) to the clipboard. Share it to let others see the exact same view." />
        <SmallButton label="PNG" onClick={() => exportImage("png")} tip="Export the current canvas as a PNG image file." />
        <SmallButton label="JPEG" onClick={() => exportImage("jpeg")} tip="Export the current canvas as a JPEG image file (smaller file size, slight quality loss)." />
        {(mode === "mandelbrot" || mode === "julia") && (<>
          <SmallButton label="Zoom In" onClick={() => zoomView(0.5)} tip="Zoom in 2x centered on the current view." />
          <SmallButton label="Zoom Out" onClick={() => zoomView(2)} tip="Zoom out 2x centered on the current view." />
        </>)}
        {mode === "mandelbrot" && <SmallButton label="Reset" onClick={() => setMbViewport({ ...DEFAULT_MB_VP })} tip="Return to the default full-set view." />}
        {mode === "julia" && <SmallButton label="Reset" onClick={() => setJuliaViewport({ ...DEFAULT_JL_VP })} tip="Return to the default Julia set viewport." />}
        <SmallButton
          label="Reset All"
          onClick={handleGlobalReset}
          tip="Restore the entire application to its default startup state."
        />
      </div>
    </div>
  );

  const getCPowerBounds = () => {
    if (mode === "mandelbrot") {
      // Burning Ship and Tricorn warp heavily, so restrict the max range
      if (mbEquation === 1 || mbEquation === 2) return { min: 0.1, max: 4, step: 0.01 };
      // Standard Mandelbrot handles high fractional powers cleanly
      return { min: 0.01, max: 10, step: 0.01 };
    }
    // Julia set bounds
    return { min: 0.01, max: 10, step: 0.01 };
  };

  const cBounds = getCPowerBounds();

  return (
    <div style={{
      position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
      display: "flex", flexDirection: "column",
      background: COLORS.bg, color: COLORS.text, fontFamily: FONT_UI, overflow: "hidden",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Fira+Code:wght@400;500&display=swap');
        html, body, #root { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: ${COLORS.panel}; }
        ::-webkit-scrollbar-thumb { background: ${COLORS.panelBorder}; border-radius: 2px; }
        input[type="range"] { -webkit-appearance: none; appearance: none; background: ${COLORS.inputBorder}; border-radius: 2px; outline: none; }
        input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 10px; height: 10px; border-radius: 2px; background: ${COLORS.accent}; cursor: pointer; }
        select { cursor: pointer; }
        select option { background: ${COLORS.panel}; color: ${COLORS.text}; }
      `}</style>

      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "8px 16px", borderBottom: `1px solid ${COLORS.panelBorder}`,
        background: COLORS.panel, height: "44px", minHeight: "44px", flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <span style={{ fontFamily: FONT_BODY, fontSize: "15px", fontWeight: 500, color: COLORS.white, letterSpacing: "2px" }}>CORALS</span>
          <span style={{ fontFamily: FONT_UI, fontSize: "10px", color: COLORS.textDim, letterSpacing: "0.5px" }}>fractal explorer, © shauryashaurya</span>
        </div>
        <div style={{ display: "flex", gap: "4px" }}>
          <TabButton label="Mandelbrot" active={mode === "mandelbrot"} onClick={() => setMode("mandelbrot")}
            tip="Escape-time fractal: iterate z = z^2 + c for each pixel. The set of points that do not escape." />
          <TabButton label="Julia Sets" active={mode === "julia"} onClick={() => setMode("julia")}
            tip="Same iteration as Mandelbrot but c is fixed and z varies. Each point in the Mandelbrot set defines a Julia set." />
          <TabButton label="Recursive" active={mode === "recursive"} onClick={() => setMode("recursive")}
            tip="Classic recursive fractals: Sierpinski triangle, Koch snowflake, dragon curve, fractal tree. Direct visualisations of recursive functions." />
          <TabButton label="L-Systems" active={mode === "lsystem"} onClick={() => setMode("lsystem")}
            tip="Fractals defined by string rewriting rules and turtle graphics. A formal grammar approach to fractal generation." />
        </div>
      </div>

      <div style={{ display: "flex", flex: 1, minHeight: 0, overflow: "hidden" }}>
        {renderSidebar()}
        <div ref={containerRef}
          onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp}
          onMouseLeave={() => { handleMouseUp(); setMouseCoord(null); }}
          onWheel={handleWheel} onDoubleClick={handleDoubleClick}
          style={{
            flex: 1, minWidth: 0, position: "relative", overflow: "hidden",
            background: isEscapeMode ? COLORS.canvasBg : "#fafaf8",
            cursor: isEscapeMode ? (isDragging ? "grabbing" : "crosshair") : "default",
          }}>
          <canvas ref={glCanvasRef} style={{
            position: "absolute", top: 0, left: 0, width: "100%", height: "100%",
            display: (isEscapeMode && !cpuRenderActive) ? "block" : "none",
          }} />
          <canvas ref={canvasRef} style={{
            position: "absolute", top: 0, left: 0, width: "100%", height: "100%",
            display: (!isEscapeMode || cpuRenderActive) ? "block" : "none",
          }} />
          {isEscapeMode && (<>
            <CoordDisplay coord={mouseCoord} renderTime={renderTime} />
            <ViewportDisplay viewport={mode === "mandelbrot" ? mbViewport : juliaViewport} />
            <div style={{
              position: "absolute", top: "8px", right: "8px", fontFamily: FONT_UI, fontSize: "10px",
              color: COLORS.textDim, background: "rgba(6,6,11,0.8)", padding: "6px 10px",
              borderRadius: "3px", pointerEvents: "none", zIndex: 10, lineHeight: "1.7",
              border: `1px solid ${COLORS.panelBorder}`,
            }}>
              <span style={{ color: COLORS.accent }}>Drag</span> to pan &nbsp;
              <span style={{ color: COLORS.accent }}>Scroll</span> to zoom &nbsp;
              <span style={{ color: COLORS.accent }}>Double-click</span> to zoom in
            </div>
          </>)}
          {!isEscapeMode && renderTime != null && (
            <div style={{
              position: "absolute", bottom: "8px", left: "8px", fontFamily: FONT_BODY, fontSize: "11px",
              color: "#888", background: "rgba(250,250,248,0.85)", padding: "4px 8px", borderRadius: "2px",
              pointerEvents: "none",
            }}>{renderTime}ms</div>
          )}
          <CodePanel visible={showCode} snippetKeys={getCodeKeys()} onClose={() => setShowCode(false)} />
        </div>
      </div>
    </div>
  );
}
