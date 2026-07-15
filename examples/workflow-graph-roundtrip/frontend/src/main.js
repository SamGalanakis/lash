import '@fontsource-variable/hanken-grotesk';
import '@fontsource-variable/spline-sans-mono';
import '@xyflow/svelte/dist/style.css';
import './styles/global.css';
import { mount } from 'svelte';
import App from './App.svelte';

const app = mount(App, { target: document.getElementById('app') });

export default app;
