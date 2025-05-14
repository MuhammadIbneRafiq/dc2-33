import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, SendHorizontal, ChevronDown, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface Message {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: Date;
}

interface PoliceChatProps {
  selectedLSOA?: string | null;
  selectedAllocation?: any | null;
}

const PoliceChat: React.FC<PoliceChatProps> = ({ selectedLSOA, selectedAllocation }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [inputMessage, setInputMessage] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      sender: 'assistant',
      text: "Ello, ello, ello! I'm your Police Chat Assistant. How can I help with the residential burglary data? Select an area on the map or ask about EMMIE scores!",
      timestamp: new Date()
    }
  ]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Update when selected LSOA changes
  useEffect(() => {
    if (selectedLSOA && isOpen) {
      const newMessage: Message = {
        id: Date.now().toString(),
        sender: 'assistant',
        text: `You've selected LSOA ${selectedLSOA}. This area shows elevated residential burglary risk. Would you like me to provide the EMMIE recommendations for this location?`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, newMessage]);
    }
  }, [selectedLSOA, isOpen]);

  // Update when allocation data changes
  useEffect(() => {
    if (selectedAllocation && isOpen) {
      const newMessage: Message = {
        id: Date.now().toString(),
        sender: 'assistant',
        text: `The resource allocation has been updated! ${selectedAllocation.vehiclePatrols} vehicle and ${selectedAllocation.footPatrols} foot patrols have been assigned. The predicted effectiveness rating is ${selectedAllocation.avgEffectiveness.toFixed(1)}%. Shall I go through the details, guv?`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, newMessage]);
    }
  }, [selectedAllocation, isOpen]);

  const handleSendMessage = () => {
    if (!inputMessage.trim()) return;
    
    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      sender: 'user',
      text: inputMessage,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    
    // Process the query and respond
    setTimeout(() => {
      const responseText = generateResponse(inputMessage);
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'assistant',
        text: responseText,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, assistantMessage]);
    }, 1000);
  };

  const generateResponse = (query: string): string => {
    const lowercaseQuery = query.toLowerCase();
    
    // Check for EMMIE related questions
    if (lowercaseQuery.includes('emmie')) {
      return "EMMIE stands for Effects, Mechanisms, Moderators, Implementation and Economic impacts. It's our framework for evidence-based policing. Each intervention gets a score based on these five dimensions. A bit like how we rate our tea breaks, but more scientific, innit!";
    }
    
    // Check for burglary related questions
    if (lowercaseQuery.includes('burglary') || lowercaseQuery.includes('burglar')) {
      return "Right, so our residential burglary prediction model uses historical data, geographic profiling, and socioeconomic factors. It's not just guesswork - it's like having a detective's intuition backed by proper data. The correlation between our predictions and actual burglaries last quarter was 89%. Not too shabby!";
    }
    
    // Check for allocation questions
    if (lowercaseQuery.includes('allocation') || lowercaseQuery.includes('deploy') || lowercaseQuery.includes('resource')) {
      return "Police resource allocation now uses CPTED (Crime Prevention Through Environmental Design) principles. Officers are deployed based on factors like natural surveillance, territorial reinforcement, access control, maintenance, and activity support, ensuring maximum impact in reducing burglary risk. This is a smarter, evidence-based approach to patrol planning.";
    }
    
    // Check for map questions
    if (lowercaseQuery.includes('map') || lowercaseQuery.includes('area') || lowercaseQuery.includes('location')) {
      return "The map shows residential burglary risk by area. Red spots are high risk - that's where we need more visibility. Yellow is moderate risk. If you click on an area, I can tell you more about the local factors and what interventions might work best there. It's like having the local bobby's knowledge but for all of London!";
    }

    // Default responses with police humor
    const defaultResponses = [
      "I'm on the case! Let me get my notepad out... What specifically about the residential burglary forecast would you like to know?",
      "You don't need to take a statement - I already know the data inside out! What aspect of the residential burglary predictions are you curious about?",
      "Well spotted! That's a good question. Let me check the evidence... Anything specific about the residential burglary analysis you'd like me to focus on?",
      "Right then, what we've got here is a situation that requires some detective work. Could you be more specific about which data you're interested in?",
      "Copy that. I'll need backup from the database to answer properly. Could you clarify which part of the residential burglary forecast interests you?"
    ];
    
    return defaultResponses[Math.floor(Math.random() * defaultResponses.length)];
  };

  // Common questions that can be clicked
  const commonQuestions = [
    "What is EMMIE?",
    "How accurate is the burglary prediction?",
    "How does resource allocation work?",
    "What do the colors on the map mean?"
  ];

  const handleCommonQuestion = (question: string) => {
    setInputMessage(question);
    // Small delay to show the question being typed
    setTimeout(() => {
      handleSendMessage();
    }, 100);
  };

  const formatTime = (date: Date): string => {
    return date.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <>
      {/* Chat Button */}
      <div className="fixed bottom-6 right-6 z-50">
        <motion.button
          onClick={() => setIsOpen(!isOpen)}
          className="bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-full shadow-lg flex items-center justify-center"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {isOpen ? <X size={24} /> : <MessageCircle size={24} />}
        </motion.button>
      </div>

      {/* Chat Widget */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.9 }}
            className="fixed bottom-20 right-6 w-80 sm:w-96 bg-gray-900 rounded-lg shadow-xl border border-gray-700 z-50 flex flex-col"
            style={{ maxHeight: 'calc(100vh - 150px)' }}
          >
            {/* Chat Header */}
            <div className="p-4 border-b border-gray-700 bg-gradient-to-r from-blue-900/50 to-indigo-900/30 rounded-t-lg">
              <div className="flex items-center">
                <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center mr-3">
                  <MessageCircle size={16} className="text-white" />
                </div>
                <div>
                  <h3 className="text-white font-medium">Police Chat Assistant</h3>
                  <p className="text-gray-400 text-xs">Scotland Yard Support</p>
                </div>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.map((msg) => (
                <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div 
                    className={`max-w-3/4 p-3 rounded-lg ${
                      msg.sender === 'user' 
                        ? 'bg-blue-600 text-white rounded-br-none' 
                        : 'bg-gray-800 text-gray-200 rounded-bl-none'
                    }`}
                  >
                    <p className="text-sm">{msg.text}</p>
                    <p className="text-xs text-gray-400 mt-1 text-right">{formatTime(msg.timestamp)}</p>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            {/* Common Questions */}
            <div className="px-4 py-2 bg-gray-800/50 border-t border-gray-700">
              <p className="text-xs text-gray-400 mb-2">Common questions:</p>
              <div className="flex flex-wrap gap-2">
                {commonQuestions.map((question, index) => (
                  <button
                    key={index}
                    onClick={() => handleCommonQuestion(question)}
                    className="text-xs bg-gray-700 hover:bg-gray-600 text-blue-300 px-2 py-1 rounded-full transition-colors"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>

            {/* Input */}
            <div className="p-3 border-t border-gray-700">
              <div className="flex items-center bg-gray-800 rounded-lg px-3 py-2">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                  placeholder="Type your question..."
                  className="bg-transparent flex-1 text-white outline-none text-sm"
                />
                <button
                  onClick={handleSendMessage}
                  disabled={!inputMessage.trim()}
                  className={`ml-2 ${
                    inputMessage.trim() ? 'text-blue-400 hover:text-blue-300' : 'text-gray-500'
                  }`}
                >
                  <SendHorizontal size={18} />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default PoliceChat;
