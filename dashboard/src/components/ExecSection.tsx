/**
 * One numbered section of the executive narrative. Shared between the
 * Executive Dashboard and Executive Recommendation pages, which used to be
 * one file (`ExecutiveSummary.tsx`) — split so each answers a narrower
 * question, but the section chrome itself didn't need to change.
 */

import type { ReactNode } from 'react';

interface ExecSectionProps {
  index: number;
  eyebrow: string;
  title: string;
  children: ReactNode;
}

export function ExecSection({ index, eyebrow, title, children }: ExecSectionProps) {
  return (
    <section className="section exec-section">
      <div className="section__head">
        <p className="eyebrow">
          {index}. {eyebrow}
        </p>
        <h2 className="section__title">{title}</h2>
      </div>
      {children}
    </section>
  );
}
