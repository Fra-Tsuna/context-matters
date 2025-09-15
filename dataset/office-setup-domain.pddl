(define (domain office-setup-domain)

  (:requirements
    :strips
    :typing
    :derived-predicates
  )

  (:types
    room locatable - object
    robot grabbable container - locatable 
  )

  (:predicates
    (at ?something - locatable ?where - room)
    (is-holding ?who - robot ?something - grabbable)
    (is-free ?who - robot)
    (inside ?what - grabbable ?where - container)
  )

  ;; True iff no object is inside the container
  (:derived (empty ?m - container)
    (forall (?x - object) (not (inside ?x ?m)))
  )

  (:action move_to
    :parameters (?who - robot ?from - room ?to - room)
    :precondition (and (at ?who ?from))
    :effect (and (not (at ?who ?from)) (at ?who ?to))
  )

  (:action grab
    :parameters (?who - robot ?what - grabbable ?where - room)
    :precondition (and (at ?who ?where) (at ?what ?where) (is-free ?who))
    :effect (and (not (at ?what ?where)) (is-holding ?who ?what) (not (is-free ?who)))
  )

  (:action drop
    :parameters (?who - robot ?what - grabbable ?where - room)
    :precondition (and (at ?who ?where) (is-holding ?who ?what))
    :effect (and (not (is-holding ?who ?what)) (is-free ?who) (at ?what ?where))
  )

  (:action push
    :parameters (?who - robot ?what - container ?from - room ?to - room)
    :precondition (and
      (at ?who ?from)
      (at ?what ?from)
      (empty ?what)       
    )
    :effect (and
      (not (at ?who ?from))
      (not (at ?what ?from))
      (at ?who ?to)
      (at ?what ?to)
    )
  )

  (:action take_out
      :parameters (?who - robot ?what - grabbable ?from - container ?where - room)
      :precondition (and (at ?who ?where) (at ?from ?where) (inside ?what ?from) (is-free ?who))
      :effect (and (not (inside ?what ?from)) (is-holding ?who ?what) (not (is-free ?who)))
  ) ;; no need to manage (empty ...) — it’s derived and updates automatically


  (:action put_in
    :parameters (?who - robot ?what - grabbable ?in - container ?where - room)
    :precondition (and (at ?who ?where)(at ?in ?where)(is-holding ?who ?what))
    :effect (and (not (is-holding ?who ?what))(inside ?what ?in)(is-free ?who))
  )
)
