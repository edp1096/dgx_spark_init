import { mount } from 'svelte'
import App from './App.svelte'
import './app.css'
import './forms.css'
import './recognition.css'
import './jobs.css'
import './settings.css'
import './enhancer.css'
import './lora.css'
import './assistant.css'
import './tags.css'
import './theme.css'

const savedTheme = localStorage.getItem('spark-media-theme')
const initialTheme = savedTheme === 'light' ? 'light' : 'dark'
document.documentElement.dataset.theme = initialTheme
document.documentElement.style.colorScheme = initialTheme

mount(App, { target: document.getElementById('app') })
