import { cn } from "../../lib/utils";

export const BentoGrid = ({
  className,
  children,
}) => {
  return (
    <div
      className={cn(
        "grid md:auto-rows-[18rem] grid-cols-1 md:grid-cols-3 gap-4 max-w-7xl mx-auto",
        className
      )}
    >
      {children}
    </div>
  );
};

export const BentoGridItem = ({
  className,
  title,
  description,
  header,
  icon,
}) => {
  return (
    <div
      className={cn(
        "rounded-2xl overflow-hidden row-span-1 border border-gold-border group/bento hover:shadow-xl transition duration-200 shadow-none p-6 bg-primary flex flex-col justify-between",
        className
      )}
    >
      {header}
      <div className="group-hover/bento:translate-x-2 transition duration-200">
        <div className="text-secondary mb-2 mt-2">{icon}</div>
        <div className="font-sans font-bold text-text-primary mb-2 mt-2">
          {title}
        </div>
        <div className="font-sans font-normal text-text-secondary text-sm">
          {description}
        </div>
      </div>
    </div>
  );
};
