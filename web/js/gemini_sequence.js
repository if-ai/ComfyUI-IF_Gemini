import { app } from "/scripts/app.js";

// Extension for handling Gemini multimodal sequence generation in ComfyUI
app.registerExtension({
    name: "Comfy.GeminiSequence",
    
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
            // Store the original onNodeCreated function so we can call it
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            
            // Enhance the node creation process to add sequence functionality
            nodeType.prototype.onNodeCreated = function() {
                // Call the original function if it exists
                if (originalOnNodeCreated) {
                    originalOnNodeCreated.apply(this, arguments);
                }
                
                // Create a container for sequence output
                this.sequenceContainer = document.createElement("div");
                this.sequenceContainer.className = "if-gemini-sequence-container";
                this.sequenceContainer.style.display = "none";
                this.sequenceContainer.style.maxHeight = "400px";
                this.sequenceContainer.style.overflowY = "auto";
                this.sequenceContainer.style.padding = "10px";
                this.sequenceContainer.style.backgroundColor = "#1a1a1a";
                this.sequenceContainer.style.border = "1px solid #333";
                this.sequenceContainer.style.borderRadius = "5px";
                this.sequenceContainer.style.marginTop = "10px";
                
                // Append container to the DOM (it will be positioned later)
                document.body.appendChild(this.sequenceContainer);
                
                // Create a wrapper for controls
                this.sequenceControlsWrapper = document.createElement("div");
                this.sequenceControlsWrapper.className = "if-gemini-sequence-controls";
                this.sequenceControlsWrapper.style.display = "none";
                this.sequenceControlsWrapper.style.padding = "5px";
                this.sequenceControlsWrapper.style.backgroundColor = "#2a2a2a";
                this.sequenceControlsWrapper.style.borderRadius = "5px";
                this.sequenceControlsWrapper.style.marginTop = "5px";
                this.sequenceControlsWrapper.style.textAlign = "right";
                
                // Create a button to toggle the sequence panel
                this.toggleSequenceButton = document.createElement("button");
                this.toggleSequenceButton.textContent = "Show Sequence Panel";
                this.toggleSequenceButton.style.padding = "3px 8px";
                this.toggleSequenceButton.style.backgroundColor = "#444";
                this.toggleSequenceButton.style.border = "none";
                this.toggleSequenceButton.style.borderRadius = "3px";
                this.toggleSequenceButton.style.color = "#fff";
                this.toggleSequenceButton.style.cursor = "pointer";
                this.toggleSequenceButton.style.display = "none";
                
                // Add click handler for toggle button
                this.toggleSequenceButton.addEventListener("click", () => {
                    if (this.sequenceContainer.style.display === "none") {
                        this.showSequencePanel();
                    } else {
                        this.hideSequencePanel();
                    }
                });
                
                // Create a button to clear the sequence panel
                this.clearSequenceButton = document.createElement("button");
                this.clearSequenceButton.textContent = "Clear";
                this.clearSequenceButton.style.padding = "3px 8px";
                this.clearSequenceButton.style.backgroundColor = "#555";
                this.clearSequenceButton.style.border = "none";
                this.clearSequenceButton.style.borderRadius = "3px";
                this.clearSequenceButton.style.color = "#fff";
                this.clearSequenceButton.style.cursor = "pointer";
                this.clearSequenceButton.style.marginLeft = "5px";
                
                // Add click handler for clear button
                this.clearSequenceButton.addEventListener("click", () => {
                    this.clearSequencePanel();
                });
                
                // Add buttons to controls wrapper
                this.sequenceControlsWrapper.appendChild(this.toggleSequenceButton);
                this.sequenceControlsWrapper.appendChild(this.clearSequenceButton);
                
                // Append controls to body (will be positioned with the node)
                document.body.appendChild(this.sequenceControlsWrapper);
                
                // Setup method to show the sequence panel
                this.showSequencePanel = function() {
                    this.sequenceContainer.style.display = "block";
                    this.toggleSequenceButton.textContent = "Hide Sequence Panel";
                    this.positionSequencePanel();
                };
                
                // Setup method to hide the sequence panel
                this.hideSequencePanel = function() {
                    this.sequenceContainer.style.display = "none";
                    this.toggleSequenceButton.textContent = "Show Sequence Panel";
                };
                
                // Setup method to clear the sequence panel
                this.clearSequencePanel = function() {
                    this.sequenceContainer.innerHTML = "";
                };
                
                // Method to position the sequence panel relative to the node
                this.positionSequencePanel = function() {
                    if (!this.sequenceContainer) return;
                    
                    const nodeRect = this.domElement.getBoundingClientRect();
                    const verticalOffset = 10;
                    
                    this.sequenceContainer.style.position = "absolute";
                    this.sequenceContainer.style.top = (nodeRect.bottom + verticalOffset) + "px";
                    this.sequenceContainer.style.left = nodeRect.left + "px";
                    this.sequenceContainer.style.width = (nodeRect.width * 1.5) + "px";
                    this.sequenceContainer.style.zIndex = 1000;
                    
                    // Position controls below the node as well
                    this.sequenceControlsWrapper.style.position = "absolute";
                    this.sequenceControlsWrapper.style.top = (nodeRect.bottom + verticalOffset - 35) + "px";
                    this.sequenceControlsWrapper.style.left = nodeRect.left + "px";
                    this.sequenceControlsWrapper.style.zIndex = 1001;
                };
                
                // Handle operation mode changes to show/hide sequence controls
                const operationModeWidget = this.widgets.find(w => w.name === "operation_mode");
                if (operationModeWidget) {
                    // Store original callback to chain it
                    const originalCallback = operationModeWidget.callback;
                    
                    operationModeWidget.callback = (v) => {
                        if (originalCallback) {
                            originalCallback.call(this, v);
                        }
                        
                        if (v === "generate_sequence") {
                            this.toggleSequenceButton.style.display = "inline-block";
                            this.sequenceControlsWrapper.style.display = "block";
                        } else {
                            this.toggleSequenceButton.style.display = "none";
                            this.sequenceControlsWrapper.style.display = "none";
                            this.hideSequencePanel();
                        }
                    };
                }
            };
            
            // Function to add elements to the sequence panel
            nodeType.prototype.addToSequencePanel = function(content, type = "text") {
                if (!this.sequenceContainer) return;
                
                const element = document.createElement("div");
                element.className = `if-gemini-sequence-item if-gemini-sequence-${type}`;
                element.style.marginBottom = "10px";
                element.style.padding = "8px";
                element.style.borderRadius = "5px";
                
                if (type === "text") {
                    element.style.backgroundColor = "#2a2a2a";
                    element.textContent = content;
                } else if (type === "image") {
                    element.style.textAlign = "center";
                    const img = document.createElement("img");
                    img.src = `data:${content.mimeType || "image/png"};base64,${content.data}`;
                    img.style.maxWidth = "100%";
                    img.style.borderRadius = "5px";
                    element.appendChild(img);
                } else if (type === "error") {
                    element.style.backgroundColor = "#3a2020";
                    element.style.color = "#ff7070";
                    element.textContent = content;
                } else if (type === "status") {
                    element.style.backgroundColor = "#202a3a";
                    element.style.color = "#70a0ff";
                    element.textContent = content;
                }
                
                this.sequenceContainer.appendChild(element);
                this.sequenceContainer.scrollTop = this.sequenceContainer.scrollHeight;
                
                // Make sure panel is shown
                if (this.sequenceContainer.style.display === "none") {
                    this.showSequencePanel();
                }
            };
            
            // Enhance the computeSize method to handle sequence panel positioning
            const originalComputeSize = nodeType.prototype.computeSize;
            nodeType.prototype.computeSize = function() {
                const size = originalComputeSize ? originalComputeSize.apply(this, arguments) : this.size;
                // Reposition the sequence panel if it exists and is visible
                if (this.sequenceContainer && this.sequenceContainer.style.display !== "none") {
                    setTimeout(() => this.positionSequencePanel(), 10);
                }
                return size;
            };
            
            // Enhance the onDragMove method to move the sequence panel with the node
            const originalOnDragMove = nodeType.prototype.onDragMove;
            nodeType.prototype.onDragMove = function(evt) {
                if (originalOnDragMove) {
                    originalOnDragMove.apply(this, arguments);
                }
                
                // Reposition the sequence panel when the node is dragged
                if (this.sequenceContainer) {
                    this.positionSequencePanel();
                }
            };
            
            // Register WebSocket handlers
            
            // Initialize sequence generation
            app.registerExtension({
                name: "Comfy.GeminiSequenceMessages",
                init() {
                    // Handler for sequence initialization message
                    api.addEventListener("if-gemini-sequence-init", (e) => {
                        const data = e.detail;
                        const node = app.graph.getNodeById(data.node_id);
                        if (!node) return;
                        
                        // Clear any existing content
                        if (node.clearSequencePanel) {
                            node.clearSequencePanel();
                        }
                        
                        // Add initialization message
                        if (node.addToSequencePanel) {
                            node.addToSequencePanel("Initializing sequence generation...", "status");
                        }
                    });
                    
                    // Handler for text messages
                    api.addEventListener("if-gemini-sequence-text", (e) => {
                        const data = e.detail;
                        const node = app.graph.getNodeById(data.node_id);
                        if (!node || !node.addToSequencePanel) return;
                        
                        node.addToSequencePanel(data.text, "text");
                    });
                    
                    // Handler for image messages
                    api.addEventListener("if-gemini-sequence-image", (e) => {
                        const data = e.detail;
                        const node = app.graph.getNodeById(data.node_id);
                        if (!node || !node.addToSequencePanel) return;
                        
                        node.addToSequencePanel({
                            data: data.image_b64,
                            mimeType: data.mime_type
                        }, "image");
                    });
                    
                    // Handler for error messages
                    api.addEventListener("if-gemini-sequence-error", (e) => {
                        const data = e.detail;
                        const node = app.graph.getNodeById(data.node_id);
                        if (!node || !node.addToSequencePanel) return;
                        
                        node.addToSequencePanel(data.message, "error");
                    });
                    
                    // Handler for completion messages
                    api.addEventListener("if-gemini-sequence-complete", (e) => {
                        const data = e.detail;
                        const node = app.graph.getNodeById(data.node_id);
                        if (!node || !node.addToSequencePanel) return;
                        
                        node.addToSequencePanel(data.message, "status");
                    });
                }
            });
        }
    }
}); 