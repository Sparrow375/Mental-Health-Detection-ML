import { motion } from 'framer-motion';

export default function AnimatedSection({ children, className = '', delay = 0 }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.85, y: 40, filter: 'blur(10px)' }}
      whileInView={{ opacity: 1, scale: 1, y: 0, filter: 'blur(0px)' }}
      viewport={{ once: false, margin: '-50px' }}
      transition={{ 
        type: 'spring', 
        stiffness: 150, 
        damping: 15, 
        delay,
        mass: 0.8 
      }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
