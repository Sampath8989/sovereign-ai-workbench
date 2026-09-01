/**
 * DeliverableViewer.test.tsx
 *
 * Tests:
 * - File path in response renders a download link
 * - No file path renders nothing (empty state)
 * - Multiple file paths render multiple links
 * - Path traversal in filenames is safe
 * - Correct download URL is constructed
 */
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import DeliverableViewer from '../../components/DeliverableViewer';

describe('DeliverableViewer', () => {
  it('renders download link when response contains a .docx file path', () => {
    render(
      <DeliverableViewer response="Generated file at workspace/outputs/report.docx" />
    );

    const link = screen.getByRole('link', { name: /report\.docx/i });
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute('href', expect.stringContaining('report.docx'));
  });

  it('renders download link for .xlsx files', () => {
    render(
      <DeliverableViewer response="Spreadsheet saved to workspace/outputs/data.xlsx" />
    );

    expect(screen.getByRole('link', { name: /data\.xlsx/i })).toBeInTheDocument();
  });

  it('renders download link for .pptx files', () => {
    render(
      <DeliverableViewer response="Presentation created at workspace/outputs/slides.pptx" />
    );

    expect(screen.getByRole('link', { name: /slides\.pptx/i })).toBeInTheDocument();
  });

  it('renders nothing when response has no file path', () => {
    const { container } = render(
      <DeliverableViewer response="The answer is 4." />
    );

    expect(container.textContent).toBe('');
  });

  it('renders nothing when response is empty', () => {
    const { container } = render(
      <DeliverableViewer response="" />
    );

    expect(container.textContent).toBe('');
  });

  it('renders nothing when response is null/undefined', () => {
    const { container } = render(
      <DeliverableViewer response={null as any} />
    );

    expect(container.textContent).toBe('');
  });

  it('renders multiple download links for multiple file paths', () => {
    render(
      <DeliverableViewer response="Created report.docx and data.xlsx in workspace/outputs/" />
    );

    expect(screen.getByRole('link', { name: /report\.docx/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /data\.xlsx/i })).toBeInTheDocument();
  });

  it('does NOT render download link for path traversal attempts', () => {
    const { container } = render(
      <DeliverableViewer response="File at ../../../etc/passwd.txt" />
    );

    // Should not render a link for this suspicious path
    const links = container.querySelectorAll('a');
    expect(links.length).toBe(0);
  });

  it('uses /api/download endpoint for download links', () => {
    render(
      <DeliverableViewer response="File at workspace/outputs/test.docx" />
    );

    const link = screen.getByRole('link', { name: /test\.docx/i });
    expect(link).toHaveAttribute('href', expect.stringContaining('/api/download'));
  });
});
