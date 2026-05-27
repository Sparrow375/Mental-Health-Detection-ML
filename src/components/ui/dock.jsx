import { cn } from "../../lib/utils";
import { motion } from "framer-motion";
import React from "react";

const Dock = React.forwardRef(({ className, children, ...props }, ref) => (
  <motion.div
    ref={ref}
    className={cn(
      "mx-auto flex h-14 items-center gap-4 bg-primary border border-gold-border px-4",
      className
    )}
    {...props}
  >
    {children}
  </motion.div>
));
Dock.displayName = "Dock";

const DockIcon = ({ children, className, ...props }) => {
  return (
    <motion.div
      whileHover={{ scale: 1.1, backgroundColor: 'rgba(212, 175, 55, 0.1)' }}
      whileTap={{ scale: 0.95 }}
      className={cn("flex aspect-square cursor-pointer items-center justify-center transition-colors p-2 text-text-secondary hover:text-secondary", className)}
      {...props}
    >
      {children}
    </motion.div>
  );
};
DockIcon.displayName = "DockIcon";

export { Dock, DockIcon };
