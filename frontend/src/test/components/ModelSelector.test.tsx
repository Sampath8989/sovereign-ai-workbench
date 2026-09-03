import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import ModelSelector from '../../components/ModelSelector'

describe('ModelSelector Component', () => {
  const mockOnSelect = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders Auto as the default selected model', () => {
    render(<ModelSelector selectedModel="auto" onSelectModel={mockOnSelect} />)
    expect(screen.getAllByText(/Auto/i).length).toBeGreaterThanOrEqual(1)
  })

  it('opens the dropdown menu when clicked', async () => {
    render(<ModelSelector selectedModel="auto" onSelectModel={mockOnSelect} />)
    const button = screen.getByRole('button')
    fireEvent.click(button)

    expect(screen.getByText(/Local Model Engine/i)).toBeInTheDocument()
    expect(screen.getByText(/DeepSeek R1 7B/i)).toBeInTheDocument()
    expect(screen.getByText(/Phi-4 14B/i)).toBeInTheDocument()
    expect(screen.getByText(/Qwen 2.5 Coder 7B/i)).toBeInTheDocument()
    expect(screen.getByText(/LLaVA 7B/i)).toBeInTheDocument()
  })

  it('calls onSelectModel when a model is clicked', async () => {
    render(<ModelSelector selectedModel="auto" onSelectModel={mockOnSelect} />)
    const button = screen.getByRole('button')
    fireEvent.click(button)

    const deepseekOption = screen.getByText(/DeepSeek R1 7B/i)
    fireEvent.click(deepseekOption)

    expect(mockOnSelect).toHaveBeenCalledWith('deepseek-r1-7b.gguf')
  })

  it('displays selected model name when a specific model is active', () => {
    render(
      <ModelSelector
        selectedModel="deepseek-r1-7b.gguf"
        onSelectModel={mockOnSelect}
      />
    )
    expect(screen.getByText(/DeepSeek R1 7B/i)).toBeInTheDocument()
  })

  it('shows category badges for models (REASONING, CODE, VISION, GENERAL)', async () => {
    render(<ModelSelector selectedModel="auto" onSelectModel={mockOnSelect} />)
    fireEvent.click(screen.getByRole('button'))

    expect(screen.getAllByText('REASONING').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('CODE').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('VISION').length).toBeGreaterThanOrEqual(1)
  })
})
