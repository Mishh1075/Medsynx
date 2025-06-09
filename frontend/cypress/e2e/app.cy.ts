describe('Medsynx E2E Tests', () => {
  beforeEach(() => {
    // Reset database and seed test data
    cy.request('POST', '/api/test/reset-db');
    cy.visit('/');
  });

  it('should allow user registration and login', () => {
    // Register new user
    cy.visit('/register');
    cy.get('[data-testid=email-input]').type('test@example.com');
    cy.get('[data-testid=password-input]').type('password123');
    cy.get('[data-testid=confirm-password-input]').type('password123');
    cy.get('[data-testid=register-button]').click();

    // Should redirect to login
    cy.url().should('include', '/login');

    // Login
    cy.get('[data-testid=email-input]').type('test@example.com');
    cy.get('[data-testid=password-input]').type('password123');
    cy.get('[data-testid=login-button]').click();

    // Should redirect to dashboard
    cy.url().should('include', '/dashboard');
  });

  it('should handle data upload and generation flow', () => {
    // Login first
    cy.login('test@example.com', 'password123');

    // Upload data
    cy.get('[data-testid=upload-button]').click();
    cy.get('[data-testid=file-input]').attachFile('test-data.csv');
    cy.get('[data-testid=upload-submit]').click();

    // Should show upload success
    cy.get('[data-testid=upload-success]').should('be.visible');

    // Configure generation settings
    cy.get('[data-testid=epsilon-input]').type('1.0');
    cy.get('[data-testid=delta-input]').type('0.00001');
    cy.get('[data-testid=model-select]').select('dpgan');
    cy.get('[data-testid=generate-button]').click();

    // Should show generation progress
    cy.get('[data-testid=progress-bar]').should('be.visible');

    // Wait for generation to complete
    cy.get('[data-testid=generation-complete]', { timeout: 30000 }).should('be.visible');

    // Download results
    cy.get('[data-testid=download-button]').click();
    cy.readFile('cypress/downloads/synthetic_data.csv').should('exist');
  });

  it('should display privacy and utility metrics', () => {
    // Login and generate data first
    cy.login('test@example.com', 'password123');
    cy.generateSyntheticData();

    // Check privacy metrics
    cy.get('[data-testid=privacy-tab]').click();
    cy.get('[data-testid=epsilon-score]').should('be.visible');
    cy.get('[data-testid=disclosure-score]').should('be.visible');

    // Check utility metrics
    cy.get('[data-testid=utility-tab]').click();
    cy.get('[data-testid=similarity-score]').should('be.visible');
    cy.get('[data-testid=correlation-score]').should('be.visible');
  });

  it('should handle batch processing', () => {
    // Login first
    cy.login('test@example.com', 'password123');

    // Upload multiple files
    cy.get('[data-testid=batch-upload]').click();
    cy.get('[data-testid=file-input]')
      .attachFile(['data1.csv', 'data2.csv', 'data3.csv']);
    cy.get('[data-testid=batch-submit]').click();

    // Should show batch progress
    cy.get('[data-testid=batch-progress]').should('be.visible');

    // Wait for all jobs to complete
    cy.get('[data-testid=batch-complete]', { timeout: 60000 }).should('be.visible');

    // Check results for each file
    cy.get('[data-testid=batch-results]').children().should('have.length', 3);
  });

  it('should allow model parameter tuning', () => {
    // Login first
    cy.login('test@example.com', 'password123');

    // Go to parameter tuning page
    cy.get('[data-testid=tuning-tab]').click();

    // Configure parameters
    cy.get('[data-testid=batch-size-input]').type('64');
    cy.get('[data-testid=epochs-input]').type('100');
    cy.get('[data-testid=learning-rate-input]').type('0.001');

    // Start tuning
    cy.get('[data-testid=tune-button]').click();

    // Should show tuning progress
    cy.get('[data-testid=tuning-progress]').should('be.visible');

    // Wait for tuning to complete
    cy.get('[data-testid=tuning-complete]', { timeout: 120000 }).should('be.visible');

    // Check tuning results
    cy.get('[data-testid=tuning-results]').should('be.visible');
    cy.get('[data-testid=best-params]').should('be.visible');
  });
}); 