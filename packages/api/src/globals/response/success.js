/**
 * @description Success Class for success responses
 */
class Success {
  constructor(message, status = 200) {
    this.message = message;
    this.status = status;
    this.success = true;
  }
}

export { Success };
