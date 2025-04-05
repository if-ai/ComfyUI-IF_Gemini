// gemini_node.js - Combined logic for GeminiNode UI and sequence display
import { app } from "/scripts/app.js";

// Main extension registration
app.registerExtension({
    name: "Comfy.GeminiNode", // Main extension for the node

    // Setup: Wait for ComfyUI to fully initialize before setting up
    async setup() {
        console.log("Setting up Gemini Node extension...");
        
        // Wait for ComfyUI to initialize
        const maxAttempts = 10;
        let attempts = 0;
        while ((!app.ui?.settings?.store || !window.api) && attempts < maxAttempts) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            attempts++;
        }
        
        if (!window.api) {
            console.error("API not available after waiting. Some Gemini features may not work properly.");
            return;
        }
        
        // Set up WebSocket event listeners
        try {
            this.setupSequenceListeners();
            console.log("Gemini Node event listeners set up successfully.");
        } catch (error) {
            console.error("Failed to set up Gemini event listeners:", error);
        }
    },
    
    // Separate method for setting up sequence-related event listeners
    setupSequenceListeners() {
        if (!window.api) {
            console.warn("API not available, skipping sequence listeners setup");
            return;
        }
        
        // --- Sequence Listeners ---
        window.api.addEventListener("if-gemini-sequence-init", (event) => {
            try {
                const detail = event.detail;
                const node = app.graph.getNodeById(detail.node_id);
                if (node?.type === "GeminiNode" && node.clearSequencePanel) {
                    node.clearSequencePanel();
                    node.addToSequencePanel(detail.message || "Initializing sequence...", "status");
                }
            } catch (error) {
                console.error("Error handling sequence-init event:", error);
            }
        });

        window.api.addEventListener("if-gemini-sequence-text", (event) => {
            try {
                const detail = event.detail;
                const node = app.graph.getNodeById(detail.node_id);
                if (node?.type === "GeminiNode" && node.addToSequencePanel) {
                    node.addToSequencePanel(detail.text, detail.error ? "error" : "text");
                }
            } catch (error) {
                console.error("Error handling sequence-text event:", error);
            }
        });

        window.api.addEventListener("if-gemini-sequence-image", (event) => {
            try {
                const detail = event.detail;
                const node = app.graph.getNodeById(detail.node_id);
                if (node?.type === "GeminiNode" && node.addToSequencePanel) {
                    node.addToSequencePanel({
                        data: detail.image_b64,
                        mimeType: detail.mime_type || "image/png"
                    }, "image");
                }
            } catch (error) {
                console.error("Error handling sequence-image event:", error);
            }
        });

        window.api.addEventListener("if-gemini-sequence-error", (event) => {
            try {
                const detail = event.detail;
                const node = app.graph.getNodeById(detail.node_id);
                if (node?.type === "GeminiNode" && node.addToSequencePanel) {
                    node.addToSequencePanel(detail.message, "error");
                }
            } catch (error) {
                console.error("Error handling sequence-error event:", error);
            }
        });

        window.api.addEventListener("if-gemini-sequence-complete", (event) => {
            try {
                const detail = event.detail;
                const node = app.graph.getNodeById(detail.node_id);
                if (node?.type === "GeminiNode" && node.addToSequencePanel) {
                    node.addToSequencePanel(detail.message, "status");
                }
            } catch (error) {
                console.error("Error handling sequence-complete event:", error);
            }
        });
    },

    // Modify node definition
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeminiNode") {
            // Store the original methods to call them later
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            const originalComputeSize = nodeType.prototype.computeSize;
            const originalOnExecuted = nodeType.prototype.onExecuted;
            const originalOnExecutionStart = nodeType.prototype.onExecutionStart;
            const originalOnDragMove = nodeType.prototype.onDragMove;
            const originalOnDrawForeground = nodeType.prototype.onDrawForeground;

            // Enhance node creation
            nodeType.prototype.onNodeCreated = function() {
                try {
                    if (originalOnNodeCreated) {
                        originalOnNodeCreated.apply(this, arguments);
                    }

                    // --- Standard UI elements (prompt size, buttons, model list) ---
                    const promptWidget = this.widgets.find(w => w.name === "prompt");
                    if (promptWidget) { 
                        promptWidget.computeSize = function(width) {
                            return [width, 120]; // Make textarea taller
                        };
                    }
                    
                    // Store model widget reference for later updates
                    this.modelWidget = this.widgets.find(w => w.name === "model_name");

                    // Add update models button
                    const updateModelsBtn = this.addWidget("button", "Update Models List", null, () => {
                        this.updateGeminiModels();
                    });
                    updateModelsBtn.serialize = false;

                    // Add verify API key button
                    const verifyApiKeyBtn = this.addWidget("button", "Verify API Key", null, () => {
                        const externalApiKeyWidget = this.widgets.find(w => w.name === "external_api_key");
                        const apiKey = externalApiKeyWidget ? externalApiKeyWidget.value : "";

                        // Add UI feedback
                        const statusWidget = this.widgets.find(w => w.name === "api_key_status");
                        if (statusWidget) {
                            statusWidget.value = "Checking API key...";
                        } else {
                            this.addWidget("text", "api_key_status", "Checking API key...");
                        }
                        
                        // Send request to verify API key
                        fetch("/gemini/check_api_key", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ api_key: apiKey })
                        })
                        .then(response => response.json())
                        .then(data => {
                            const statusWidget = this.widgets.find(w => w.name === "api_key_status");
                            if (data.status === "success") {
                                if (statusWidget) {
                                    statusWidget.value = "✅ " + data.message;
                                } else {
                                    this.addWidget("text", "api_key_status", "✅ " + data.message);
                                }
                                // After successful verification, update models list
                                this.updateGeminiModels();
                            } else {
                                if (statusWidget) {
                                    statusWidget.value = "❌ " + data.message;
                                } else {
                                    this.addWidget("text", "api_key_status", "❌ " + data.message);
                                }
                            }
                        })
                        .catch(error => {
                            console.error("Error verifying API key:", error);
                            const statusWidget = this.widgets.find(w => w.name === "api_key_status");
                            if (statusWidget) {
                                statusWidget.value = "❌ Error communicating with server";
                            } else {
                                this.addWidget("text", "api_key_status", "❌ Error communicating with server");
                            }
                        });
                    });
                    
                    // Configure button
                    verifyApiKeyBtn.serialize = false;
                    
                    // Add initial status widget
                    const statusWidget = this.addWidget("text", "api_key_status", "API key not verified");
                    statusWidget.serialize = false;

                    // Try to update models list on node creation
                    setTimeout(() => {
                        this.updateGeminiModels();
                    }, 1000);

                    // Listen for changes to the API key input
                    const apiKeyWidget = this.widgets.find(w => w.name === "external_api_key");
                    if (apiKeyWidget) {
                        const originalCallback = apiKeyWidget.callback;
                        apiKeyWidget.callback = (v) => {
                            if (originalCallback) {
                                originalCallback.call(this, v);
                            }
                            // If API key has at least 10 characters, try to update models
                            if (v && v.length >= 10) {
                                setTimeout(() => {
                                    this.updateGeminiModels();
                                }, 1000);
                            }
                        };
                    }

                    // --- Sequence Panel Setup ---
                    // Create simple text area for sequence output (simpler than DOM widget)
                    this.sequenceContainer = document.createElement("div");
                    this.sequenceContainer.className = "if-gemini-sequence-container";
                    this.sequenceContainer.style.cssText = `
                        display: none; max-height: 300px; overflow-y: auto; margin-top: 10px;
                        padding: 8px; border: 1px solid #333; background-color: #222;
                        color: #ccc; font-size: 12px; line-height: 1.4; border-radius: 4px;
                    `;
                    
                    // Append to node's DOM element
                    if (this.domElement) {
                        this.domElement.appendChild(this.sequenceContainer);
                    }

                    // Add sequence control buttons
                    const toggleSequenceBtn = this.addWidget("button", "Toggle Sequence Panel", null, () => {
                        if (!this.sequenceContainer) return;
                        
                        if (this.sequenceContainer.style.display === "none") {
                            this.sequenceContainer.style.display = "block";
                        } else {
                            this.sequenceContainer.style.display = "none";
                        }
                        this.setDirtyCanvas(true, true);
                        this.graph.setDirtyCanvas(true, true);
                    });
                    toggleSequenceBtn.serialize = false;
                    
                    const clearSequenceBtn = this.addWidget("button", "Clear Sequence", null, () => {
                        this.clearSequencePanel();
                    });
                    clearSequenceBtn.serialize = false;

                    // --- Sequence Helper Methods ---
                    this.addToSequencePanel = function(content, type = "text") {
                        if (!this.sequenceContainer) return;
                        this.sequenceContainer.style.display = "block"; // Ensure visible

                        const element = document.createElement("div");
                        element.style.cssText = `margin-bottom: 8px; padding: 6px; border-radius: 3px;`;

                        if (type === "text") {
                            element.style.backgroundColor = "#282828";
                            element.textContent = content;
                            element.style.whiteSpace = "pre-wrap";
                        } else if (type === "image") {
                            element.style.textAlign = "center";
                            try {
                                const img = document.createElement("img");
                                img.src = `data:${content.mimeType || "image/png"};base64,${content.data}`;
                                img.style.cssText = `max-width: 95%; border-radius: 3px; margin-top: 4px; border: 1px solid #444;`;
                                element.appendChild(img);
                            } catch (e) {
                                console.error("Error creating image in sequence panel:", e);
                                element.textContent = "[Error displaying image]";
                                element.style.color = "#ff8080";
                            }
                        } else if (type === "error") {
                            element.style.backgroundColor = "#402020";
                            element.style.color = "#ff8080";
                            element.textContent = `❌ ${content}`;
                        } else if (type === "status") {
                            element.style.backgroundColor = "#203040";
                            element.style.color = "#80b0ff";
                            element.textContent = content;
                        }

                        this.sequenceContainer.appendChild(element);
                        this.sequenceContainer.scrollTop = this.sequenceContainer.scrollHeight; // Scroll down
                        this.setDirtyCanvas(true, true); // Trigger redraw
                    };

                    this.clearSequencePanel = function() {
                        if (this.sequenceContainer) {
                            this.sequenceContainer.innerHTML = '';
                            this.sequenceContainer.style.display = 'none'; // Hide it
                        }
                        this.setDirtyCanvas(true, true); // Trigger redraw
                    };

                    // Show/hide sequence controls based on operation mode
                    const operationModeWidget = this.widgets.find(w => w.name === "operation_mode");
                    if (operationModeWidget) {
                        const originalModeCallback = operationModeWidget.callback;
                        operationModeWidget.callback = (v) => {
                            if (originalModeCallback) {
                                originalModeCallback.call(this, v);
                            }
                            
                            // Show sequence controls only in sequence mode
                            const isSequenceMode = v === "generate_sequence";
                            toggleSequenceBtn.style.display = isSequenceMode ? "" : "none";
                            clearSequenceBtn.style.display = isSequenceMode ? "" : "none";
                            
                            // Hide panel when switching away from sequence mode
                            if (!isSequenceMode && this.sequenceContainer) {
                                this.sequenceContainer.style.display = "none";
                            }
                        };
                        
                        // Initial state
                        const isSequenceMode = operationModeWidget.value === "generate_sequence";
                        toggleSequenceBtn.style.display = isSequenceMode ? "" : "none";
                        clearSequenceBtn.style.display = isSequenceMode ? "" : "none";
                    }
                } catch (error) {
                    console.error("Error in GeminiNode onNodeCreated:", error);
                }
            }; // End of onNodeCreated

            // Override computeSize to handle sequence panel sizing
            nodeType.prototype.computeSize = function() {
                let size = [200, 100]; // Default size
                if (originalComputeSize) {
                    size = originalComputeSize.apply(this, arguments);
                }
                return size;
            };

            // Override execution lifecycle
            nodeType.prototype.onExecuted = function(message) {
                try {
                    if (originalOnExecuted) {
                        originalOnExecuted.apply(this, arguments);
                    }
                    
                    const operationMode = this.widgets.find(w => w.name === "operation_mode")?.value;
                    
                    // Store text output for standard display
                    if (message && message.text && operationMode !== "generate_sequence") {
                        this.generated_text = message.text;
                        this.setDirtyCanvas(true, true);
                    }
                } catch (error) {
                    console.error("Error in GeminiNode onExecuted:", error);
                }
            };

            nodeType.prototype.onExecutionStart = function() {
                try {
                    const operationMode = this.widgets.find(w => w.name === "operation_mode")?.value;
                    // Only auto-clear in sequence mode
                    if (operationMode === "generate_sequence") {
                        this.clearSequencePanel?.();
                    }
                    
                    // Clear previous text output
                    this.generated_text = "";
                    this.setDirtyCanvas(true, true);
                    
                    if (originalOnExecutionStart) {
                        return originalOnExecutionStart.apply(this, arguments);
                    }
                    return true;
                } catch (error) {
                    console.error("Error in GeminiNode onExecutionStart:", error);
                    return true; // Allow execution to continue
                }
            };

            // Handle node movement for sequence panel
            nodeType.prototype.onDragMove = function(e) {
                try {
                    if (originalOnDragMove) {
                        originalOnDragMove.apply(this, arguments);
                    }
                    // Ensure sequence panel stays with node
                    if (this.sequenceContainer) {
                        this.sequenceContainer.style.position = "relative";
                    }
                } catch (error) {
                    console.error("Error in GeminiNode onDragMove:", error);
                }
            };

            // Function to update Gemini models
            nodeType.prototype.updateGeminiModels = function() {
                if (!this.modelWidget) {
                    return;
                }

                const externalApiKeyWidget = this.widgets.find(w => w.name === "external_api_key");
                const apiKey = externalApiKeyWidget ? externalApiKeyWidget.value : "";

                // Get models from the backend
                fetch("/gemini/get_models", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ external_api_key: apiKey })
                })
                .then(response => response.json())
                .then(models => {
                    if (models && models.length > 0) {
                        // Store current model selection
                        const currentModel = this.modelWidget.value;
                        
                        // Update the model widget options
                        this.modelWidget.options.values = models;
                        
                        // Try to maintain the current model if it exists in new list
                        if (models.includes(currentModel)) {
                            this.modelWidget.value = currentModel;
                        } else {
                            // Default to first model in the list
                            this.modelWidget.value = models[0];
                        }
                        
                        // Trigger a property change and update the UI
                        const modelChangedEvent = this.modelWidget.options?.onchange;
                        if (modelChangedEvent) {
                            modelChangedEvent.call(this.modelWidget, this.modelWidget.value);
                        }
                        
                        // Add models info to status widget
                        const statusWidget = this.widgets.find(w => w.name === "api_key_status");
                        if (statusWidget && statusWidget.value && statusWidget.value.includes("✅")) {
                            statusWidget.value += ` (${models.length} models available)`;
                        }
                        
                        this.setDirtyCanvas(true, true);
                    }
                })
                .catch(error => {
                    console.error("Error updating Gemini models:", error);
                });
            };

            // Add custom drawing to show generated text below the node
            nodeType.prototype.onDrawForeground = function(ctx) {
                try {
                    if (originalOnDrawForeground) {
                        originalOnDrawForeground.apply(this, arguments);
                    }
                    
                    // Display generated text when available
                    if (this.generated_text) {
                        const margin = 10;
                        const textX = this.pos[0] + margin;
                        const textY = this.pos[1] + this.size[1] + 20;
                        const maxWidth = this.size[0] - margin * 2;
                        
                        ctx.save();
                        ctx.font = "12px Arial";
                        ctx.fillStyle = "#CCC";
                        this.wrapText(ctx, this.generated_text, textX, textY, maxWidth, 16);
                        ctx.restore();
                    }
                } catch (error) {
                    console.error("Error in GeminiNode onDrawForeground:", error);
                }
            };

            // Add text wrapping helper function
            nodeType.prototype.wrapText = function(ctx, text, x, y, maxWidth, lineHeight) {
                if (!text) return;
                
                try {
                    const words = text.split(' ');
                    let line = '';
                    let posY = y;
                    const maxLines = 10; // Limit number of preview lines
                    let lineCount = 0;

                    for (const word of words) {
                        if (lineCount >= maxLines) {
                            ctx.fillText("...", x, posY);
                            break;
                        }
                        
                        const testLine = line + word + ' ';
                        const metrics = ctx.measureText(testLine);
                        const testWidth = metrics.width;

                        if (testWidth > maxWidth && line !== '') {
                            ctx.fillText(line, x, posY);
                            line = word + ' ';
                            posY += lineHeight;
                            lineCount++;
                        } else {
                            line = testLine;
                        }
                    }
                    
                    if (lineCount < maxLines) {
                        ctx.fillText(line, x, posY);
                    }
                } catch (error) {
                    console.error("Error in GeminiNode wrapText:", error);
                }
            };
        } // End of if (nodeData.name === "GeminiNode")
    } // End of beforeRegisterNodeDef
}); // End of app.registerExtension 