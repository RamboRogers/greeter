let ws = null;
let lastFrameTime = 0;
const MIN_FRAME_INTERVAL = 30; // milliseconds between frame updates

function connectWebSocket() {
    if (ws) {
        ws.close();
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = () => {
        console.log('WebSocket connected');
        document.getElementById('camera-status').textContent = 'Connected';
        document.getElementById('camera-status').className = 'status-connected';
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'frame') {
            const currentTime = performance.now();
            if (currentTime - lastFrameTime >= MIN_FRAME_INTERVAL) {
                updateFrame(data.data);
                lastFrameTime = currentTime;

                // Update status including FPS
                const statusElement = document.getElementById('camera-status');
                const status = data.status;
                statusElement.textContent = `Connected (${status.fps} FPS)`;
                statusElement.className = 'status-connected';
            }
        }
    };

    ws.onclose = () => {
        console.log('WebSocket closed');
        document.getElementById('camera-status').textContent = 'Disconnected';
        document.getElementById('camera-status').className = 'status-disconnected';
        setTimeout(connectWebSocket, 1000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        document.getElementById('camera-status').textContent = 'Error';
        document.getElementById('camera-status').className = 'status-error';
    };
}

function toggleCamera() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
    } else {
        connectWebSocket();
    }
}

function updateFrame(frameData) {
    const img = document.getElementById('camera-feed');
    img.src = 'data:image/jpeg;base64,' + frameData;
}

function refreshLogs() {
    fetch('/api/logs')
        .then(response => response.json())
        .then(data => {
            const logsContainer = document.getElementById('logs-container');
            logsContainer.innerHTML = ''; // Clear existing logs

            data.logs.reverse().forEach(log => {
                const logEntry = document.createElement('div');
                logEntry.className = 'log-entry';

                // Create timestamp element
                const timestamp = document.createElement('span');
                timestamp.className = 'log-timestamp';
                timestamp.textContent = log.timestamp;

                // Create message element
                const message = document.createElement('span');
                message.className = 'log-message';
                message.textContent = log.message;

                // Add elements to log entry
                logEntry.appendChild(timestamp);
                logEntry.appendChild(message);
                logsContainer.appendChild(logEntry);
            });

            // Scroll to top to see newest logs
            logsContainer.scrollTop = 0;
        })
        .catch(error => {
            console.error('Error fetching logs:', error);
        });
}

function showUnknowns() {
    const modal = document.getElementById('unknowns-modal');
    const grid = document.querySelector('.unknown-faces-grid');

    // Clear existing content
    grid.innerHTML = '';

    // Fetch unknown faces
    fetch('/api/unknown-faces')
        .then(response => response.json())
        .then(faces => {
            faces.forEach(face => {
                const div = document.createElement('div');
                div.className = 'unknown-face-item';
                div.innerHTML = `
                    <img src="/unknown_faces/${face}" alt="Unknown face">
                `;
                div.onclick = () => div.classList.toggle('selected');
                grid.appendChild(div);
            });
        });

    modal.style.display = 'block';
}

function saveSelectedFaces() {
    const newName = document.getElementById('new-name').value;
    if (!newName || !newName.match(/^[A-Za-z]+\.[A-Za-z]+$/)) {
        alert('Please enter a valid name in format: First.Last');
        return;
    }

    const selected = document.querySelectorAll('.unknown-face-item.selected img');
    const files = Array.from(selected).map(img => img.src.split('/').pop());

    fetch('/api/save-known-faces', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            name: newName,
            files: files
        })
    })
    .then(response => response.json())
    .then(result => {
        if (result.success) {
            alert('Faces saved successfully!');
            showUnknowns();  // Refresh the grid
        } else {
            alert('Error saving faces: ' + result.error);
        }
    });
}

// Close modal when clicking the X or outside
document.querySelector('.close').onclick = () => {
    document.getElementById('unknowns-modal').style.display = 'none';
}

window.onclick = (event) => {
    const modal = document.getElementById('unknowns-modal');
    if (event.target == modal) {
        modal.style.display = 'none';
    }
}

// Initialize WebSocket connection and start log refresh interval when page loads
document.addEventListener('DOMContentLoaded', function() {
    connectWebSocket();
    // Start auto-refresh of logs every 500ms
    setInterval(refreshLogs, 500);
});

async function deleteAllUnknowns() {
    if (confirm('Are you sure you want to delete all unknown faces?')) {
        try {
            const response = await fetch('/api/unknown-faces/delete-all', {
                method: 'POST'
            });
            const data = await response.json();

            if (data.success) {
                // Refresh the unknown faces display
                showUnknowns();
            } else {
                alert('Failed to delete unknown faces');
            }
        } catch (error) {
            console.error('Error deleting unknown faces:', error);
            alert('Error deleting unknown faces');
        }
    }
}

async function showKnowns() {
    const modal = document.getElementById('knowns-modal');
    const grid = modal.querySelector('.known-faces-grid');
    grid.innerHTML = 'Loading...';

    try {
        const response = await fetch('/api/known-faces');
        const data = await response.json();

        grid.innerHTML = '';

        // Group faces by person name (everything before _face)
        const peopleGroups = {};
        data.faces.forEach(face => {
            const personName = face.split('_face')[0];
            if (!peopleGroups[personName]) {
                peopleGroups[personName] = [];
            }
            peopleGroups[personName].push(face);
        });

        // Create a card for each person
        Object.entries(peopleGroups).forEach(([person, faces]) => {
            const card = document.createElement('div');
            card.className = 'known-face-item';

            // Use the first face as thumbnail
            card.innerHTML = `
                <img src="/known_faces/${faces[0]}" alt="${person}">
                <div class="known-face-name">${person}</div>
                <div class="known-face-count">${faces.length} photos</div>
                <button onclick="deletePerson('${person}')" class="delete-person-btn">Delete Person</button>
            `;

            grid.appendChild(card);
        });
    } catch (error) {
        console.error('Error loading known faces:', error);
        grid.innerHTML = 'Error loading known faces';
    }

    modal.style.display = 'block';

    // Close button functionality
    modal.querySelector('.close').onclick = () => {
        modal.style.display = 'none';
    }

    window.onclick = (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    }
}

async function deletePerson(personName) {
    if (confirm(`Are you sure you want to delete all photos of ${personName}?`)) {
        try {
            const response = await fetch('/api/known-faces/delete-person', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ person: personName })
            });

            const data = await response.json();
            if (data.success) {
                showKnowns(); // Refresh the display
            } else {
                alert('Failed to delete person');
            }
        } catch (error) {
            console.error('Error deleting person:', error);
            alert('Error deleting person');
        }
    }
}

async function showActions() {
    const modal = document.getElementById('actions-modal');

    try {
        // Load existing configuration
        const response = await fetch('/api/actions/config');
        const config = await response.json();

        if (config.success) {
            // Unknown webhook config
            document.getElementById('unknown-webhook').value = config.config.unknown_webhook?.url || '';
            document.getElementById('unknown-method').value = config.config.unknown_webhook?.method || 'POST';
            document.getElementById('unknown-headers').value = config.config.unknown_webhook?.headers ?
                JSON.stringify(config.config.unknown_webhook.headers, null, 2) : '';
            document.getElementById('unknown-body').value = config.config.unknown_webhook?.body_template || '';

            // Known webhook config
            document.getElementById('known-webhook').value = config.config.known_webhook?.url || '';
            document.getElementById('known-method').value = config.config.known_webhook?.method || 'POST';
            document.getElementById('known-headers').value = config.config.known_webhook?.headers ?
                JSON.stringify(config.config.known_webhook.headers, null, 2) : '';
            document.getElementById('known-body').value = config.config.known_webhook?.body_template || '';

            // Other webhook config
            document.getElementById('other-webhook').value = config.config.other_webhook?.url || '';
            document.getElementById('other-method').value = config.config.other_webhook?.method || 'POST';
            document.getElementById('other-headers').value = config.config.other_webhook?.headers ?
                JSON.stringify(config.config.other_webhook.headers, null, 2) : '';
            document.getElementById('other-body').value = config.config.other_webhook?.body_template || '';
        }
    } catch (error) {
        console.error('Error loading action configuration:', error);
    }

    modal.style.display = 'block';

    // Close button functionality
    modal.querySelector('.close').onclick = () => {
        modal.style.display = 'none';
    }

    window.onclick = (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    }
}

async function saveActionConfig() {
    try {
        const config = {
            unknown_webhook: {
                url: document.getElementById('unknown-webhook').value.trim(),
                method: document.getElementById('unknown-method').value,
                headers: parseJsonSafely(document.getElementById('unknown-headers').value),
                body_template: document.getElementById('unknown-body').value.trim()
            },
            known_webhook: {
                url: document.getElementById('known-webhook').value.trim(),
                method: document.getElementById('known-method').value,
                headers: parseJsonSafely(document.getElementById('known-headers').value),
                body_template: document.getElementById('known-body').value.trim()
            },
            other_webhook: {
                url: document.getElementById('other-webhook').value.trim(),
                method: document.getElementById('other-method').value,
                headers: parseJsonSafely(document.getElementById('other-headers').value),
                body_template: document.getElementById('other-body').value.trim()
            }
        };

        const response = await fetch('/api/actions/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });

        const result = await response.json();
        if (result.success) {
            alert('Configuration saved successfully');
            document.getElementById('actions-modal').style.display = 'none';
        } else {
            alert('Failed to save configuration: ' + (result.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error saving action configuration:', error);
        alert('Error saving configuration');
    }
}

async function showDatabase() {
    const modal = document.getElementById('database-modal');
    const tbody = document.querySelector('#database-table tbody');

    try {
        const response = await fetch('/api/database');
        const data = await response.json();

        tbody.innerHTML = '';

        data.forEach(row => {
            const tr = document.createElement('tr');
            const imageCell = row.image_path ?
                `<td><img src="${row.image_path}" class="face-thumbnail" onclick="showFullImage('${row.image_path}')"></td>` :
                '<td>-</td>';

            tr.innerHTML = `
                <td>${row.timestamp}</td>
                ${imageCell}
                <td>${row.person_name}</td>
                <td>${row.category}</td>
                <td>${row.sentiment || '-'}</td>
                <td>${row.confidence.toFixed(3)}</td>
                <td>${row.action}</td>
            `;
            tbody.appendChild(tr);
        });

        modal.style.display = 'block';
    } catch (error) {
        console.error('Failed to fetch database:', error);
    }
}

function showFullImage(imagePath) {
    const modal = document.getElementById('image-modal');
    const fullSizeImage = document.getElementById('full-size-image');
    fullSizeImage.src = imagePath;
    modal.style.display = 'block';
}

// Add to existing modal close handlers
document.querySelectorAll('.modal .close').forEach(closeBtn => {
    closeBtn.onclick = function() {
        this.closest('.modal').style.display = 'none';
    }
});

function switchTab(tabName) {
    // Remove active class from all buttons
    document.querySelectorAll('.action-tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Add active class to clicked button using data-tab attribute
    const activeBtn = document.querySelector(`.action-tab-btn[data-tab="${tabName}"]`);
    if (activeBtn) {
        activeBtn.classList.add('active');
    }

    // Load the configuration for this tab
    loadWebhookConfig(tabName);
}

// Add click event listeners to tabs
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.action-tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            switchTab(btn.dataset.tab);
        });
    });

    // Set first tab as active by default
    const firstTab = document.querySelector('.action-tab-btn');
    if (firstTab) {
        firstTab.classList.add('active');
        switchTab('unknown'); // Set default tab
    }
});

// Helper function to update form fields
function updateFormFields(tabName) {
    // Update URL placeholder
    const urlInput = document.querySelector('.webhook-url');
    if (urlInput) {
        urlInput.placeholder = `https://your-webhook-url/${tabName}`;
    }

    // Load existing configuration for the selected tab
    loadWebhookConfig(tabName);
}

// Add these functions to handle loading and saving action configurations

async function loadWebhookConfig(tabName) {
    try {
        const response = await fetch('/api/actions/config');
        const data = await response.json();

        if (data.success && data.config) {
            const config = data.config[tabName] || {};

            // Update form fields
            document.querySelector('.method-select').value = config.method || 'POST';
            document.querySelector('.webhook-url').value = config.url || '';
            document.querySelector('.headers-input').value = config.headers || '';
            document.querySelector('.body-input').value = config.body || '';
        }
    } catch (error) {
        console.error('Error loading webhook configuration:', error);
    }
}

async function saveActionConfig() {
    try {
        // Get the active tab
        const activeTab = document.querySelector('.action-tab-btn.active').dataset.tab;

        // Get current configurations for all tabs
        const response = await fetch('/api/actions/config');
        const data = await response.json();
        const existingConfig = data.success ? data.config : {};

        // Get form values for the active tab
        const config = {
            ...existingConfig,
            [activeTab]: {
                method: document.querySelector('.method-select').value,
                url: document.querySelector('.webhook-url').value,
                headers: document.querySelector('.headers-input').value,
                body: document.querySelector('.body-input').value
            }
        };

        // Save the configuration
        const saveResponse = await fetch('/api/actions/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });

        const saveResult = await saveResponse.json();
        if (saveResult.success) {
            alert('Configuration saved successfully!');
        } else {
            alert('Error saving configuration: ' + saveResult.error);
        }
    } catch (error) {
        console.error('Error saving webhook configuration:', error);
        alert('Error saving configuration');
    }
}

async function uploadFace(file) {
    if (!file) return;

    console.log('Starting upload of file:', file.name); // Debug log

    const formData = new FormData();
    formData.append('file', file);

    try {
        console.log('Sending file to server...'); // Debug log
        const response = await fetch('/api/upload-face', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        console.log('Server response:', result); // Debug log

        if (result.success) {
            alert(`Successfully uploaded and processed ${result.message}`);
            showUnknowns();  // Refresh the grid
        } else {
            alert('Error uploading face: ' + result.error);
        }
    } catch (error) {
        console.error('Error uploading face:', error);
        alert('Error uploading face');
    }
}
