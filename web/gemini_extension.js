// gemini_extension.js - Main entry point for Gemini Node extension
import { app } from "/scripts/app.js";
import "./js/gemini_node.js";

// Main extension wrapper
app.registerExtension({
    name: "ComfyUI.IF_Gemini.Main",
    
    async init() {
        try {
            console.log("Initializing Gemini Node extension...");
            
            // Check if API is available (for WebSocket communications)
            if (!window.api) {
                console.warn("ComfyUI API not detected. Some Gemini features may not work properly.");
            }
        } catch (error) {
            console.error("Error initializing Gemini Node extension:", error);
        }
    }
}); 