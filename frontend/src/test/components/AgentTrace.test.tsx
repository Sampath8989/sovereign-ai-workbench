/**
 * AgentTrace.test.tsx
 *
 * Tests the redesigned timeline-based AgentTrace component.
 */
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import AgentTrace from '../../components/AgentTrace';

describe('AgentTrace', () => {
  it('renders trace steps with type labels', () => {
    const trace = [
      'Planner: Decomposed into 2 step(s)',
      'Executor: file_io.read()',
      'Executor: llm.summarize()',
      'Retriever: Found 3 sources',
      'Verifier: Grounding check passed',
      'Synthesizer: Generated final response',
    ];

    const { container } = render(<AgentTrace trace={trace} />);

    // Expand first
    fireEvent.click(screen.getByRole('button'));

    // Check type labels are rendered (multiple Executor labels expected)
    const plannerLabels = screen.getAllByText('Planner');
    expect(plannerLabels.length).toBeGreaterThanOrEqual(1);

    const executorLabels = screen.getAllByText('Executor');
    expect(executorLabels.length).toBe(2);

    // Check step count badge shows 6
    expect(screen.getByText('6')).toBeInTheDocument();

    // Check step content is present
    const text = container.textContent || '';
    expect(text).toContain('file_io.read()');
    expect(text).toContain('llm.summarize()');
    expect(text).toContain('Found 3 sources');
  });

  it('handles empty trace array — shows placeholder message', () => {
    render(<AgentTrace trace={[]} />);

    fireEvent.click(screen.getByRole('button'));
    expect(screen.getByText(/send a message/i)).toBeInTheDocument();
  });

  it('handles undefined trace — shows placeholder message', () => {
    render(<AgentTrace trace={undefined} />);

    fireEvent.click(screen.getByRole('button'));
    expect(screen.getByText(/send a message/i)).toBeInTheDocument();
  });

  it('renders single step trace', () => {
    const trace = ['Synthesizer: Generated final response'];

    const { container } = render(<AgentTrace trace={trace} />);

    fireEvent.click(screen.getByRole('button'));
    expect(screen.getByText('Synthesizer')).toBeInTheDocument();
    expect(screen.getByText('1')).toBeInTheDocument();

    const text = container.textContent || '';
    expect(text).toContain('Generated final response');
  });

  it('renders long trace arrays without truncation', () => {
    const trace = Array.from({ length: 20 }, (_, i) => `Step ${i + 1}: Action ${i + 1}`);

    const { container } = render(<AgentTrace trace={trace} />);

    expect(screen.getByText('20')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button'));
    const text = container.textContent || '';
    expect(text).toContain('Step 20: Action 20');
  });

  it('toggle collapse/expand works', () => {
    const trace = ['Planner: Decomposed', 'Synthesizer: Generated'];

    const { container } = render(<AgentTrace trace={trace} />);

    // Steps should be hidden initially (collapsed by default)
    let text = container.textContent || '';
    expect(text).not.toContain('Decomposed');

    // Click toggle to expand
    const toggleButton = screen.getByRole('button');
    fireEvent.click(toggleButton);

    // Steps should be visible when expanded (label + content split across elements)
    text = container.textContent || '';
    expect(text).toContain('Planner');
    expect(text).toContain('Decomposed');

    // Click toggle again to collapse
    fireEvent.click(toggleButton);

    // Steps should be hidden again
    text = container.textContent || '';
    expect(text).not.toContain('Decomposed');
  });
});
