// gemini_node.js - Minimal implementation for GeminiNode
import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Comfy.IFGeminiNode",
    
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
        // Only apply to our IF Gemini Node
        if (nodeData.name === "IFGeminiNode") {
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

                // Find and hide original sequential_generation widget
                const originalSequentialWidget = this.widgets.find(w => w.name === "sequential_generation");
                let originalSequentialValue = false; // Default value
                
                if (originalSequentialWidget) {
                    originalSequentialValue = originalSequentialWidget.value; // Get current value
                    originalSequentialWidget.type = "hidden"; // Hide the original widget
                    console.log(`GeminiNode ${this.id}: Hiding original sequential_generation widget.`);
                } else {
                    console.warn(`GeminiNode ${this.id}: Could not find original sequential_generation widget to hide.`);
                }

                // Find a good position for the toggle - before batch_count
                const batchCountWidget = this.widgets.find(w => w.name === "batch_count");
                let insertIndex = batchCountWidget ? this.widgets.indexOf(batchCountWidget) : -1;
                if (insertIndex === -1) {
                    insertIndex = this.widgets.length; // Fallback to end of widgets
                }

                // Add custom toggle widget with temporary name to avoid conflict
                const toggleWidgetTempName = "sequential_generation_toggle_ui";
                const sequentialToggle = this.addWidget(
                    "toggle",
                    toggleWidgetTempName,
                    originalSequentialValue,
                    (value, widget, node) => {
                        // Update the hidden original widget's value
                        const hiddenWidget = node.widgets.find(w => w.name === "sequential_generation" && w.type === "hidden");
                        if (hiddenWidget) {
                            hiddenWidget.value = value;
                            console.log(`GeminiNode ${node.id}: Hidden sequential_generation value set to ${value}`);
                        } else {
                            console.error(`GeminiNode ${node.id}: Could not find hidden sequential_generation widget to update value!`);
                        }

                        // Update batch_count label for better UX
                        const batchWidget = node.widgets.find(w => w.name === "batch_count");
                        if (batchWidget) {
                            const labelElement = batchWidget.inputEl?.previousElementSibling;
                            if (labelElement) {
                                labelElement.textContent = value ? "Sequence Steps" : "Batch Count";
                            }
                            node.setDirtyCanvas(true, true);
                        }
                    },
                    { on: "SEQUENCE", off: "BATCH" }
                );
                sequentialToggle.serialize = false; // Don't serialize the UI toggle
                
                // Rename the toggle for display purposes
                sequentialToggle.name = "sequential_generation";
                
                // No need to reposition since Python order now matches UI order
                // The widget is already in the correct position
                console.log(`GeminiNode ${this.id}: Sequential toggle is in correct position`)

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

                // Listen for changes to operation mode to filter models
                const operationModeWidget = this.widgets.find(w => w.name === "operation_mode");
                if (operationModeWidget) {
                    const originalOpCallback = operationModeWidget.callback;
                    operationModeWidget.callback = (v) => {
                        if (originalOpCallback) {
                            originalOpCallback.call(this, v);
                        }
                        // Update models when operation mode changes
                        setTimeout(() => {
                            this.updateGeminiModels();
                        }, 100);
                    };
                }
            };

            // Function to update Gemini models based on operation mode
            nodeType.prototype.updateGeminiModels = function() {
                if (!this.modelWidget) {
                    return;
                }

                const externalApiKeyWidget = this.widgets.find(w => w.name === "external_api_key");
                const operationModeWidget = this.widgets.find(w => w.name === "operation_mode");
                const apiKey = externalApiKeyWidget ? externalApiKeyWidget.value : "";
                const operationMode = operationModeWidget ? operationModeWidget.value : "analysis";

                // Get models from the backend
                fetch("/gemini/get_models", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ external_api_key: apiKey })
                })
                .then(response => response.json())
                .then(allModels => {
                    if (allModels && allModels.length > 0) {
                        // Filter models based on operation mode
                        let filteredModels = this.filterModelsByOperation(allModels, operationMode);
                        
                        // Store current model selection
                        const currentModel = this.modelWidget.value;
                        
                        // Update the model widget options
                        this.modelWidget.options.values = filteredModels;
                        
                        // Try to maintain the current model if it exists in filtered list
                        if (filteredModels.includes(currentModel)) {
                            this.modelWidget.value = currentModel;
                        } else {
                            // Default to gemini-2.5-flash or first model in the list
                            const defaultModel = filteredModels.includes("gemini-2.5-flash") 
                                ? "gemini-2.5-flash" 
                                : filteredModels[0];
                            this.modelWidget.value = defaultModel;
                        }
                        
                        // Trigger a property change and update the UI
                        const modelChangedEvent = this.modelWidget.options?.onchange;
                        if (modelChangedEvent) {
                            modelChangedEvent.call(this.modelWidget, this.modelWidget.value);
                        }
                        
                        // Add models info to status widget
                        const statusWidget = this.widgets.find(w => w.name === "api_key_status");
                        if (statusWidget && statusWidget.value && statusWidget.value.includes("✅")) {
                            statusWidget.value = `✅ API key is valid (${filteredModels.length}/${allModels.length} models for ${operationMode})`;
                        }
                        
                        this.setDirtyCanvas(true, true);
                    }
                })
                .catch(error => {
                    console.error("Error updating Gemini models:", error);
                });
            };

            // Function to filter models based on operation mode
            nodeType.prototype.filterModelsByOperation = function(allModels, operationMode) {
                const imageCapableModels = [
                    "gemini-2.5-flash-image-preview",
                    "gemini-2.5-flash", 
                    "gemini-2.5-flash-002"
                ];
                
                const textModels = allModels.filter(model => !model.includes("image"));
                const multimodalModels = allModels; // All models support multimodal
                
                switch(operationMode) {
                    case "generate_images":
                        // For image generation, only show image-capable models
                        return allModels.filter(model => 
                            imageCapableModels.some(capable => model.includes(capable.split('-').slice(0, -1).join('-')) || model === capable)
                        );
                    
                    case "analysis":
                    case "generate_text": 
                        // For text analysis, show all models but prioritize text-focused ones
                        return allModels;
                    
                    default:
                        // Default: show all available models
                        return allModels;
                }
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