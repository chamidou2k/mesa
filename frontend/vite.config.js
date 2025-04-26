import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
export default ({ mode }) => {
    console.log('Vite Mode:', mode)
    console.log('Loaded ENV:', loadEnv(mode, process.cwd()))
    return defineConfig({
	plugins: [react()],
	server: {
	    port: 9000,
	    host: false,
	},
    })
}
