import React from 'react';
import type { CUnit } from '../api/types';

export interface CUnitToggleProps {
  value: CUnit;
  onChange: (u: CUnit) => void;
}

/** C形表头内的即时切换：周期 | 单次弯曲（不需要重新处理） */
export function CUnitToggle({ value, onChange }: CUnitToggleProps) {
  const btn = (u: CUnit, text: string) => (
    <button
      type="button"
      data-testid={`cunit-${u}`}
      className={`glass-btn mini ${value === u ? 'active' : ''}`}
      style={{ padding: '0 6px', fontSize: 11, marginLeft: 3, opacity: value === u ? 1 : 0.55 }}
      onClick={() => onChange(u)}
    >
      {text}
    </button>
  );
  return (
    <span>
      {btn('cycle', '周期')}
      {btn('bend', '单次弯曲')}
    </span>
  );
}
