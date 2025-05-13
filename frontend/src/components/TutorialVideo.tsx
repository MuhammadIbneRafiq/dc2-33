import React from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { X } from 'lucide-react';

interface TutorialVideoProps {
  open: boolean;
  onClose: () => void;
}

const TutorialVideo: React.FC<TutorialVideoProps> = ({ open, onClose }) => {
  // Video tutorial steps
  const tutorialSteps = [
    {
      title: "Dashboard Overview",
      description: "The main dashboard shows burglary statistics, risk maps, and resource allocation tools.",
      videoUrl: "#dashboard-overview" // These would be actual URLs in a real implementation
    },
    {
      title: "Map Navigation",
      description: "Learn how to interact with the map, view hotspots, and understand risk levels.",
      videoUrl: "#map-navigation"
    },
    {
      title: "Police Resource Allocation",
      description: "How to use the resource allocation tools to optimize police deployment.",
      videoUrl: "#resource-allocation"
    },
    {
      title: "EMMIE Framework",
      description: "Understanding the evidence-based EMMIE framework scores and interpretation.",
      videoUrl: "#emmie-framework"
    },
    {
      title: "Chat Assistant",
      description: "How to use the chat assistant for help with the dashboard.",
      videoUrl: "#chat-assistant"
    }
  ];

  const [currentStep, setCurrentStep] = React.useState(0);
  
  const nextStep = () => {
    if (currentStep < tutorialSteps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onClose();
    }
  };
  
  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[800px] bg-gray-900 border border-gray-800 text-white p-0 overflow-hidden">
        <button
          onClick={onClose}
          className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-accent data-[state=open]:text-muted-foreground"
        >
          <X className="h-4 w-4 text-gray-400" />
          <span className="sr-only">Close</span>
        </button>
        
        {/* Video section - in a real application, this would be an actual video */}
        <div className="aspect-video bg-black flex items-center justify-center">
          <div className="text-center p-6">
            <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-blue-600/30 flex items-center justify-center">
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                width="40" 
                height="40" 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                className="text-blue-400"
              >
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
            </div>
            <h3 className="text-xl font-bold mb-2">{tutorialSteps[currentStep].title}</h3>
            <p className="text-gray-400">{tutorialSteps[currentStep].description}</p>
          </div>
        </div>
        
        {/* Tutorial navigation */}
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="text-sm text-gray-400">
              Step {currentStep + 1} of {tutorialSteps.length}
            </div>
            <div className="flex space-x-2">
              <Button 
                variant="outline" 
                size="sm" 
                onClick={prevStep}
                disabled={currentStep === 0}
              >
                Previous
              </Button>
              <Button 
                variant="default" 
                size="sm" 
                onClick={nextStep}
              >
                {currentStep === tutorialSteps.length - 1 ? "Finish" : "Next"}
              </Button>
            </div>
          </div>
          
          {/* Progress bar */}
          <div className="w-full bg-gray-800 h-1 rounded-full">
            <div 
              className="bg-blue-600 h-1 rounded-full" 
              style={{ width: `${((currentStep + 1) / tutorialSteps.length) * 100}%` }}
            ></div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default TutorialVideo; 