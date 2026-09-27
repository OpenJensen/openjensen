'use strict';
const button = document.querySelector('#connect');
const heading = document.querySelector('#heading');
const message = document.querySelector('#message');
button.addEventListener('click', async () => {
  button.disabled = true;
  heading.textContent = 'Checking your application…';
  message.textContent = 'Verifying the local API and web interface.';
  try {
    const result = await window.__TAURI__.core.invoke('probe_backend');
    if (result.status === 'ready') {
      heading.textContent = 'Connected';
      message.textContent = `Opening OPEN JENSEN ${result.version} at ${result.address}`;
      window.location.replace(result.address);
      return;
    }
    heading.textContent = 'Application unavailable';
    message.textContent = result.message;
  } catch {
    heading.textContent = 'Connection check failed';
    message.textContent = 'The desktop could not check the application. Retry, or restart this desktop window.';
  }
  button.disabled = false;
  button.textContent = 'Retry connection';
});
