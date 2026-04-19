import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
// tell Vite to use relative paths for your assets so that it builds correctly regardless of the sub-directory it is hosted in.
// thus allowing one to publish this as a GitHub Pages site as well.
export default defineConfig( {
  plugins: [ react() ],
  base: "./", // this line to fix GitHub Pages asset paths
} );
