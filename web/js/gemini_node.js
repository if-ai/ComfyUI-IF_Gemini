// gemini_node.js - Minimal implementation for GeminiNode
import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Comfy.GeminiNode",
    
    async setup() {
        // Wait for ComfyUI to fully initialize
        const maxAttempts = 10;
        let attempts = 0;
        while ((!app.ui?.settings?.store || !app.api) && attempts < maxAttempts) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            attempts++;
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only apply to GeminiNode
        if (nodeData.name === "GeminiNode") {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            
            // Enhance the node creation process
            nodeType.prototype.onNodeCreated = function() {
                if (originalOnNodeCreated) {
                    originalOnNodeCreated.apply(this, arguments);
                }

                // Make prompt textarea larger
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
                        const statusWidget = this.widgets.find(w => w.name === "api_key_status");
                        if (statusWidget) {
                            statusWidget.value = "❌ Error: " + error;
                        } else {
                            this.addWidget("text", "api_key_status", "❌ Error: " + error);
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
            const originalDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (originalDrawForeground) {
                    originalDrawForeground.apply(this, arguments);
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
            };

            // Add text wrapping helper function
            nodeType.prototype.wrapText = function(ctx, text, x, y, maxWidth, lineHeight) {
                if (!text) return;
                
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
            };

            // Handle execution results - capture generated text
            const originalOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                if (originalOnExecuted) {
                    originalOnExecuted.apply(this, arguments);
                }
                
                // Store the text output for display
                if (message && message.text) {
                    this.generated_text = message.text;
                    this.setDirtyCanvas(true, true);
                }
            };
        }
    }
}); 