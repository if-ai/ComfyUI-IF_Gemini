class ChatHistory:
    """Class to manage chat history for Gemini API"""
    def __init__(self):
        self.history = []
    
    def add_message(self, role, content):
        """Add a message to the history"""
        # Handle different content types gracefully
        if isinstance(content, list):
            # For lists (like images + text), just store a placeholder
            # The actual objects can't be serialized to history anyway
            if len(content) > 0 and isinstance(content[0], str):
                processed_content = content[0]
            else:
                processed_content = "[Visual content + text]"
        elif not isinstance(content, str):
            # For non-string content (like PIL images), use a placeholder
            processed_content = "[Non-text content]"
        else:
            processed_content = content
            
        self.history.append({"role": role, "content": processed_content})
    
    def get_messages_for_api(self):
        """Get messages in the format Gemini API expects"""
        # Convert our simplified history format to what Gemini expects
        result = []
        for msg in self.history:
            result.append({
                "role": msg["role"],
                "parts": [{"text": msg["content"]}]
            })
        return result
    
    def get_formatted_history(self):
        """Get a formatted string of the chat history for display"""
        formatted = ""
        for msg in self.history:
            role_display = "You" if msg["role"] == "user" else "Gemini"
            formatted += f"**{role_display}**: {msg['content']}\n\n"
        return formatted
    
    def clear(self):
        """Clear the history"""
        self.history = []