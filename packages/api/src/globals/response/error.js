/**
 * @description Custom Error class for Error handling
 */
class Error {
  constructor(message = "Internal Server Error", status = 500) {
    this.message = message;
    this.status = status;
    this.success = false;
  }
}

export { Error };
