import { render, screen } from '@testing-library/react'
import App from './App'
import { describe, it, expect } from 'vitest'

describe('App', () => {
  it('renders the header title', () => {
    render(<App />)
    const headerElement = screen.getByText(/Serve Analyzer/i)
    expect(headerElement).toBeInTheDocument()
  })

  it('renders the upload card', () => {
    render(<App />)
    const cardTitle = screen.getByText(/Upload Serve Video/i)
    expect(cardTitle).toBeInTheDocument()
  })
})
