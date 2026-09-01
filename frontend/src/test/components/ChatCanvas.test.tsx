/**
 * ChatCanvas.test.tsx
 *
 * Tests the controlled ChatCanvas component with real props from App.tsx.
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ChatCanvas from '../../components/ChatCanvas';

function renderChat(overrides: Partial<React.ComponentProps<typeof ChatCanvas>> = {}) {
  const defaultProps = {
    onSend: vi.fn(),
    response: null,
    loading: false,
    error: null,
    role: 'engineer',
    ...overrides,
  };
  return { ...defaultProps, ...render(<ChatCanvas {...defaultProps} />) };
}

describe('ChatCanvas (controlled component)', () => {
  it('shows skeleton shimmer when loading=true', () => {
    const { container } = renderChat({ loading: true });
    const skeletons = container.querySelectorAll('.skeleton');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it('disables input and send button when loading=true', () => {
    renderChat({ loading: true });
    const input = screen.getByPlaceholderText(/ask the workbench/i);
    expect(input).toBeDisabled();
    // Find the submit button by type (loading shows spinner, no 'Send' text)
    const form = input.closest('form')!;
    const buttons = form.querySelectorAll('button[type="submit"]');
    expect(buttons.length).toBe(1);
    expect(buttons[0]).toBeDisabled();
  });

  it('enables input when loading=false', () => {
    renderChat({ loading: false });
    const input = screen.getByPlaceholderText(/ask the workbench/i);
    expect(input).not.toBeDisabled();
  });

  it('calls onSend with trimmed prompt on form submit', async () => {
    const onSend = vi.fn();
    const user = userEvent.setup();
    renderChat({ onSend });

    const input = screen.getByPlaceholderText(/ask the workbench/i);
    await user.type(input, '  What is 2+2?  ');
    await user.keyboard('{Enter}');

    expect(onSend).toHaveBeenCalledWith('What is 2+2?');
  });

  it('does NOT call onSend when input is empty', async () => {
    const onSend = vi.fn();
    const user = userEvent.setup();
    renderChat({ onSend });

    const input = screen.getByPlaceholderText(/ask the workbench/i);
    await user.type(input, '   ');
    await user.keyboard('{Enter}');

    expect(onSend).not.toHaveBeenCalled();
  });

  it('does NOT call onSend when loading=true (prevents double-submit)', () => {
    const onSend = vi.fn();
    renderChat({ onSend, loading: true });
    expect(onSend).not.toHaveBeenCalled();
  });

  it('shows error message when error prop is set', () => {
    renderChat({ error: 'Backend connection failed' });
    expect(screen.getByText('Backend connection failed')).toBeInTheDocument();
  });

  it('does NOT show skeleton when loading=false and no error', () => {
    const { container } = renderChat({ loading: false, error: null });
    expect(container.querySelectorAll('.skeleton').length).toBe(0);
  });

  it('shows user message after response arrives', () => {
    const { rerender } = render(
      <ChatCanvas
        onSend={vi.fn()}
        response={null}
        loading={false}
        error={null}
        role="engineer"
      />
    );

    const input = screen.getByPlaceholderText(/ask the workbench/i);
    fireEvent.change(input, { target: { value: 'What is 2+2?' } });
    fireEvent.submit(input.closest('form')!);

    rerender(
      <ChatCanvas
        onSend={vi.fn()}
        response={{ response: 'The answer is 4.', trace: [] }}
        loading={false}
        error={null}
        role="engineer"
      />
    );

    expect(screen.getByText('What is 2+2?')).toBeInTheDocument();
    expect(screen.getByText('The answer is 4.')).toBeInTheDocument();
  });

  it('clears input after submit', async () => {
    const onSend = vi.fn();
    const user = userEvent.setup();
    renderChat({ onSend });

    const input = screen.getByPlaceholderText(/ask the workbench/i);
    await user.type(input, 'Hello');
    await user.keyboard('{Enter}');

    expect(input).toHaveValue('');
  });

  it('shows correct role in placeholder', () => {
    renderChat({ role: 'manager' });
    expect(screen.getByPlaceholderText(/manager/i)).toBeInTheDocument();
  });
});
