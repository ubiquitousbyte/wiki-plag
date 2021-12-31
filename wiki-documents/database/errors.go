package database

import "fmt"

type errCode int

const (
	errCodeInvalidModel errCode = iota + 1
	errCodeModelNotFound
	errCodeCreate
	errCodeUpdate
	errCodeDelete
)

func (e errCode) String() string {
	switch e {
	case errCodeInvalidModel:
		return "Invalid model"
	case errCodeModelNotFound:
		return "Model not found"
	case errCodeCreate:
		return "Cannot create application entity"
	case errCodeUpdate:
		return "Cannot update application entity"
	case errCodeDelete:
		return "Cannot delete application entity"
	default:
		return "Unknown error"
	}
}

var (
	ErrInvalidModel  = &dbErr{code: errCodeInvalidModel}
	ErrModelNotFound = &dbErr{code: errCodeModelNotFound}
	ErrCreate        = &dbErr{code: errCodeCreate}
	ErrUpdate        = &dbErr{code: errCodeUpdate}
	ErrDelete        = &dbErr{code: errCodeDelete}
)

type dbErr struct {
	code errCode
	err  error
}

func (e dbErr) Error() string {
	var msg string
	if e.err != nil {
		msg = e.err.Error()
	}
	return fmt.Sprintf("<DbErr:%d>: %s (%s)", e.code, e.code, msg)
}

func (e dbErr) from(err error) error {
	el := e
	el.err = err
	return el
}

func (e dbErr) Unwrap() error {
	return e.err
}

func (e dbErr) Is(other error) bool {
	m, ok := other.(*dbErr)
	if !ok {
		return false
	}

	return m.code == e.code
}
