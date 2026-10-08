import { mount } from 'svelte';
import App from './App.svelte';
import './app.css';
import './styles/light-theme.css';
import { applyTheme, storedTheme } from './lib/theme.js';

applyTheme(storedTheme(), false);

mount(App, { target: document.getElementById('app') });
