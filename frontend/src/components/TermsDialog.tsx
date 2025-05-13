import React, { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { AlertTriangle, Info, Video, MessageCircle } from 'lucide-react';
import { motion } from 'framer-motion';

interface TermsDialogProps {
  open: boolean;
  onClose: () => void;
  onAccept: () => void;
  onWatchTutorial: () => void;
}

const TermsDialog: React.FC<TermsDialogProps> = ({ 
  open, 
  onClose, 
  onAccept,
  onWatchTutorial
}) => {
  const [accepted, setAccepted] = useState(false);
  const [showPrivacyDetails, setShowPrivacyDetails] = useState(false);
  
  const handleAccept = () => {
    if (accepted) {
      // Store acceptance in localStorage
      localStorage.setItem('termsAccepted', 'true');
      onAccept();
    }
  };
  
  return (
    <Dialog open={open} onOpenChange={onClose} modal={true}>
      <DialogContent 
        className="sm:max-w-[600px] bg-gray-900 border border-gray-800 text-white shadow-xl z-[99999] !fixed !top-1/2 !left-1/2 !-translate-x-1/2 !-translate-y-1/2" 
        hideCloseButton
      >
        <DialogHeader>
          <DialogTitle className="text-xl font-bold text-white flex items-center">
            <AlertTriangle className="w-5 h-5 text-amber-500 mr-2" />
            Important: Confidential Crime Data
          </DialogTitle>
          <DialogDescription className="text-gray-400">
            Please review the terms and conditions before accessing the London Crime Sight dashboard.
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-4 my-4 text-gray-300">
          <div className="rounded-lg bg-gray-800 p-4 border border-gray-700">
            <p className="font-medium mb-2 text-amber-400">Confidentiality Notice</p>
            <p className="text-sm">
              This application contains sensitive crime data and predictive analytics that should be treated
              as confidential. The data and insights provided are for authorized personnel only and should not
              be shared outside your organization.
            </p>
          </div>
          
          <div className="rounded-lg bg-blue-900/30 p-4 border border-blue-800/50">
            <p className="font-medium mb-2 text-blue-400">Data Usage</p>
            <p className="text-sm">
              The crime forecast and EMMIE framework scores are based on statistical models and historical data.
              These predictions should be used as decision support tools and not as the sole basis for resource allocation.
            </p>
          </div>
          
          {showPrivacyDetails && (
            <motion.div 
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="rounded-lg bg-gray-800 p-4 border border-gray-700"
            >
              <p className="font-medium mb-2 text-purple-400">Privacy Policy</p>
              <ul className="text-sm list-disc pl-5 space-y-2">
                <li>All user interactions with this dashboard are logged for security purposes.</li>
                <li>The chat assistant interactions are recorded to improve the system.</li>
                <li>No personally identifiable information is collected from dashboard users.</li>
                <li>The data shown is anonymized and aggregated to protect privacy.</li>
                <li>This application complies with GDPR and UK Data Protection laws.</li>
              </ul>
            </motion.div>
          )}
          
          <Button 
            variant="link" 
            className="text-gray-400 text-sm px-0 flex items-center" 
            onClick={() => setShowPrivacyDetails(!showPrivacyDetails)}
          >
            <Info className="h-4 w-4 mr-1" />
            {showPrivacyDetails ? 'Hide' : 'Show'} detailed privacy information
          </Button>
          
          <div className="flex items-start space-x-2 pt-2">
            <Checkbox 
              id="terms" 
              checked={accepted}
              onCheckedChange={(checked) => setAccepted(checked as boolean)}
              className="mt-1"
            />
            <label
              htmlFor="terms"
              className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
            >
              I acknowledge that I am authorized to access this data and agree to treat all information as confidential in accordance with organizational policies.
            </label>
          </div>
        </div>
        
        <DialogFooter className="flex flex-col sm:flex-row sm:justify-between sm:space-x-2">
          <div className="flex space-x-2 mb-3 sm:mb-0">
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="text-blue-400 border-blue-400/50 hover:bg-blue-400/10"
              onClick={onWatchTutorial}
            >
              <Video className="mr-2 h-4 w-4" />
              Watch Tutorial
            </Button>
            
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="text-green-400 border-green-400/50 hover:bg-green-400/10"
              onClick={() => {
                onClose();
                // Target the chat button in the corner
                const chatButton = document.querySelector('.fixed.bottom-6.right-6 button');
                if (chatButton) {
                  (chatButton as HTMLButtonElement).click();
                }
              }}
            >
              <MessageCircle className="mr-2 h-4 w-4" />
              Get Help
            </Button>
          </div>
          
          <Button
            type="button"
            disabled={!accepted}
            onClick={handleAccept}
            className={accepted ? "bg-blue-600 hover:bg-blue-700" : "bg-gray-700 text-gray-400 cursor-not-allowed"}
          >
            I Agree & Continue
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default TermsDialog; 