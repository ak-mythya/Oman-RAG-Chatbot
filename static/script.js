document.addEventListener('DOMContentLoaded', () => {
    let sessionId = '';
    const chatBox = document.getElementById('chat-box');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');

    const chatButton = document.getElementById('chat-button');
    const chatOverlay = document.getElementById('chat-overlay');
    const minimizeButton = document.getElementById('minimize-button');
    const closeButton = document.getElementById('close-button');

    let lastQuestion = '';
    let lastAnswer = '';

    chatButton.addEventListener('click', openChatOverlay);
    minimizeButton.addEventListener('click', minimizeChatOverlay);
    closeButton.addEventListener('click', closeChatOverlay);

    chatForm.addEventListener('submit', handleChatSubmit);

    function openChatOverlay() {
        chatOverlay.classList.remove('hidden');
        chatButton.classList.add('hidden');
        userInput.focus();
    }

    function minimizeChatOverlay() {
        chatOverlay.classList.add('hidden');
        chatButton.classList.remove('hidden');
    }

    function closeChatOverlay() {
        chatOverlay.classList.add('hidden');
        chatButton.classList.remove('hidden');
        sessionId = '';
        chatBox.innerHTML = '';
    }

    async function handleChatSubmit(e) {
        e.preventDefault();
        const message = userInput.value.trim();
        if (!message) return;

        lastQuestion = message;
        appendMessage('user', message);
        userInput.value = '';

        const typingIndicator = appendMessage('bot', '...');

        try {
            const startTime = performance.now();
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: message, session_id: sessionId })
            });

            if (!response.ok) {
                throw new Error(`خطأ: تعذر جلب الاستجابة.`);
            }

            const data = await response.json();
            const endTime = performance.now();
            const durationInSeconds = ((endTime - startTime) / 1000).toFixed(2);

            if (typingIndicator.parentNode === chatBox) {
                chatBox.removeChild(typingIndicator);
            }

            if (data.session_id) {
                sessionId = data.session_id;
            }

            lastAnswer = data.answer || 'لم أتمكن من العثور على إجابة.';
            appendMessage('bot', lastAnswer, durationInSeconds);

            // Show buttons if specified by backend
            if (data.show_buttons && data.button_options && data.button_options.length > 0) {
                appendExitButtons(data.button_options);
            }

        } catch (error) {
            if (typingIndicator && typingIndicator.parentNode === chatBox) {
                chatBox.removeChild(typingIndicator);
            }
            lastAnswer = 'خطأ: تعذر جلب الاستجابة.';
            appendMessage('bot', lastAnswer);
            console.error('Error:', error);
        } finally {
            userInput.focus();
        }
    }

    function appendMessage(sender, text, responseTime = '') {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);

        const messageText = document.createElement('div');
        messageText.classList.add('message-text');
        messageText.textContent = text;
        messageDiv.appendChild(messageText);

        if (responseTime && sender === 'bot') {
            const responseTimeSpan = document.createElement('span');
            responseTimeSpan.classList.add('response-time');
            responseTimeSpan.textContent = `تم الرد في ${responseTime} ثانية`;
            messageDiv.appendChild(responseTimeSpan);
        }

        chatBox.appendChild(messageDiv);
        chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: 'smooth' });

        return messageDiv;
    }

    function appendExitButtons(buttonOptions) {
        const exitMessageDiv = document.createElement('div');
        exitMessageDiv.classList.add('message', 'bot-message');

        const exitDiv = document.createElement('div');
        exitDiv.classList.add('exit-container');

        const buttonsContainer = document.createElement('div');
        buttonsContainer.classList.add('exit-buttons');

        // Create buttons dynamically based on button_options
        buttonOptions.forEach(option => {
            const button = document.createElement('button');
            button.classList.add('exit-btn');
            // Assign specific classes and translations based on the button option
            if (option === "Do you want to start over?") {
                button.classList.add('start-over-btn');
                button.textContent = 'البدء من جديد'; // Arabic translation for "Start over"
            } else if (option === "Continue") {
                button.classList.add('continue-btn');
                button.textContent = 'متابعة'; // Arabic translation for "Continue"
            } else if (option === "Yes, Please Connect the call with a human-agent") {
                button.classList.add('yes-human-agent-btn');
                button.textContent = 'نعم، يرجى توصيل المكالمة بوكيل بشري'; // Arabic translation
            } else if (option === "No, continue the chat") {
                button.classList.add('no-continue-chat-btn');
                button.textContent = 'لا، استمر في الدردشة'; // Arabic translation
            } else {
                button.textContent = option; // Fallback
            }
            button.addEventListener('click', () => handleExitChoice(button.textContent, exitDiv));
            buttonsContainer.appendChild(button);
        });

        exitDiv.appendChild(buttonsContainer);
        exitMessageDiv.appendChild(exitDiv);
        chatBox.appendChild(exitMessageDiv);
        chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: 'smooth' });
    }

    function handleExitChoice(choice, exitDiv) {
        if (choice === "البدء من جديد") { // "Start over"
            // Clear chat and start a new session
            sessionId = ''; // Reset session ID
            chatBox.innerHTML = ''; // Clear chat
            const newSessionMessage = 'لقد بدأنا جلسة جديدة. كيف يمكنني مساعدتك؟'; // "We have started a new session. How can I help you?"
            appendMessage('bot', newSessionMessage);
            userInput.focus();
        } else if (choice === "متابعة") { // "Continue"
            // Continue the conversation
            exitDiv.innerHTML = '<div class="exit-message">يرجى طرح سؤالك التالي.</div>'; // "Please ask your next question."
            userInput.focus();
        } else if (choice === "نعم، يرجى توصيل المكالمة بوكيل بشري") { // "Yes, please connect the call with a human-agent"
            exitDiv.innerHTML = '<div class="exit-message">تم توصيل مكالمتك بوكيل بشري.</div>'; // "Your call is connected to a human agent."
            userInput.focus();
        } else if (choice === "لا، استمر في الدردشة") { // "No, continue the chat"
            exitDiv.innerHTML = '<div class="exit-message">يرجى طرح سؤالك التالي.</div>'; // "Please ask your next question."
            userInput.focus();
        }
    }
});