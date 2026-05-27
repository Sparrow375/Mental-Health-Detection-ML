import { motion } from 'framer-motion';

export default function PageTransition({ children }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8, y: 50, filter: 'blur(15px)', rotateX: 10 }}
      animate={{ opacity: 1, scale: 1, y: 0, filter: 'blur(0px)', rotateX: 0 }}
      exit={{ opacity: 0, scale: 1.2, y: -50, filter: 'blur(15px)', rotateX: -10 }}
      transition={{ 
        type: 'spring',
        stiffness: 200,
        damping: 20,
        mass: 1,
        duration: 0.5 
      }}
      style={{ perspective: '1000px' }}
    >
      {children}
    </motion.div>
  );
}
