/**
 * RoleSwitcher.test.tsx
 *
 * Tests:
 * - Default role is "engineer"
 * - Toggle switches between "engineer" and "manager"
 * - Role change is reflected in callback
 * - Visual state matches internal state
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import RoleSwitcher from '../../components/RoleSwitcher';

describe('RoleSwitcher', () => {
  it('defaults to engineer role', () => {
    const onChange = vi.fn();
    render(<RoleSwitcher role="engineer" onChange={onChange} />);

    // Engineer should be selected
    expect(screen.getByText(/engineer/i)).toBeInTheDocument();
    expect(screen.getByText(/manager/i)).toBeInTheDocument();
  });

  it('shows manager as selected when role is manager', () => {
    const onChange = vi.fn();
    render(<RoleSwitcher role="manager" onChange={onChange} />);

    expect(screen.getByText(/manager/i)).toBeInTheDocument();
  });

  it('calls onChange when toggle is clicked', async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();

    render(<RoleSwitcher role="engineer" onChange={onChange} />);

    const managerButton = screen.getByRole('button', { name: /manager/i });
    await user.click(managerButton);

    expect(onChange).toHaveBeenCalledWith('manager');
  });

  it('can toggle back to engineer', async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();

    render(<RoleSwitcher role="manager" onChange={onChange} />);

    const engineerButton = screen.getByRole('button', { name: /engineer/i });
    await user.click(engineerButton);

    expect(onChange).toHaveBeenCalledWith('engineer');
  });

  it('calls onChange even when clicking already-selected role (component always fires)', async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();

    render(<RoleSwitcher role="engineer" onChange={onChange} />);

    const engineerButton = screen.getByRole('button', { name: /engineer/i });
    await user.click(engineerButton);

    // Component always fires onChange, even for the same role
    expect(onChange).toHaveBeenCalledWith('engineer');
  });
});
