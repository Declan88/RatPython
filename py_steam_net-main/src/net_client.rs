use std::sync::{Arc, Mutex};

use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyBytes, PyList},
};
use steamworks::{
    networking_messages::NetworkingMessages,
    networking_types::{NetworkingIdentity, SendFlags},
    Client, ClientManager, DistanceFilter, LobbyChatUpdate, LobbyId, LobbyKey, LobbyType,
    SingleClient, SteamId, StringFilter, StringFilterKind,
};

// Every lobby this binding creates gets tagged with this key/value, and
// every search only ever asks Steam for lobbies carrying it - the whole
// point being that this game's own lobbies are the ONLY ones a client
// of this game will ever create or see, even though Steam's matchmaking
// is otherwise a shared, global namespace (any app using the same
// Steam AppID, or a misconfigured one, could otherwise show up in an
// unfiltered request_lobby_list). Not a generic/configurable filter -
// this project only ever wants exactly this one tag, so it's a fixed
// constant rather than a parameter threaded through create_lobby/
// get_lobby_list's own Python-facing signatures.
const GAME_IDENTITY_KEY: &str = "GameIdentity";
const GAME_IDENTITY_VALUE: &str = "Rat King";

#[pyclass(unsendable)]
pub struct PySteamClient {
    client: Option<(Client, SingleClient)>,
    messages: Option<NetworkingMessages<ClientManager>>,
    cb_conn_failed: Arc<Mutex<Option<Py<PyAny>>>>,
    cb_lobby_changed: Arc<Mutex<Option<Py<PyAny>>>>,
    cb_message_recv: Arc<Mutex<Option<Py<PyAny>>>>,
    send_errors: Mutex<std::collections::HashMap<u64, u32>>,
}

#[pymethods]
impl PySteamClient {
    #[new]
    pub fn new() -> Self {
        PySteamClient {
            client: None,
            messages: None,
            cb_conn_failed: Arc::new(Mutex::new(None)),
            cb_lobby_changed: Arc::new(Mutex::new(None)),
            cb_message_recv: Arc::new(Mutex::new(None)),
            send_errors: Mutex::new(std::collections::HashMap::new()),
        }
    }

    pub fn init(&mut self, app_id: u32) -> PyResult<()> {
        match Client::init_app(app_id) {
            Ok((client, single)) => {
                let cb_lobby_changed_shared = self.cb_lobby_changed.clone();
                client.register_callback::<LobbyChatUpdate, _>(move |update| {
                    Python::with_gil(|py| {
                        if let Some(cb) = &*cb_lobby_changed_shared.lock().unwrap() {
                            if let Err(e) = cb.call1(
                                py,
                                (
                                    update.lobby.raw(),
                                    update.user_changed.raw(),
                                    update.making_change.raw(),
                                    update.member_state_change as u64,
                                ),
                            ) {
                                e.print(py);
                            }
                        }
                    });
                });

                let networking = client.networking_messages();
                let cb_connection_failed_shared = self.cb_conn_failed.clone();
                networking.session_failed_callback(move |info| {
                    Python::with_gil(|py| {
                        if let Some(cb) = &*cb_connection_failed_shared.lock().unwrap() {
                            if let Err(e) = cb.call1(
                                py,
                                (info.identity_remote().unwrap().steam_id().unwrap().raw(),),
                            ) {
                                e.print(py);
                            }
                        }
                    });
                });

                // Steam drops a peer's messages until the session is
                // accepted - sending to them accepts it implicitly, but
                // a peer that's still loading (and so hasn't sent
                // anything yet) would have its first packets lost, and
                // the other side's would too. Accept everything here;
                // the Python side only trusts senders that are in its
                // current lobby (NetworkManager.handle_data).
                networking.session_request_callback(|request| {
                    request.accept();
                });

                self.client = Some((client, single));
                self.messages = Some(networking);
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }

    pub fn deinit(&mut self) {
        self.client = None;
    }

    pub fn is_ready(&self) -> bool {
        self.client.is_some()
    }

    pub fn run_callbacks(&self) {
        if let Some(client) = &self.client {
            client.1.run_callbacks();
        }
    }

    pub fn receive_messages(&self, channel: u32, max_messages: usize) {
        if let Some(messages) = &self.messages {
            let received_messages: Vec<(u64, u32, Vec<u8>)> = messages
                .receive_messages_on_channel(channel, max_messages)
                .into_iter()
                .filter_map(|message| {
                    if let Some(steam_id_identity) = message.identity_peer().steam_id() {
                        Some((steam_id_identity.raw(), channel, message.data().to_vec()))
                    } else {
                        None
                    }
                })
                .collect();

            Python::with_gil(|py| {
                if let Some(cb) = &*self.cb_message_recv.lock().unwrap() {
                    for (steam_id, ch, data) in received_messages {
                        if let Err(e) = cb.call1(py, (steam_id, ch, PyBytes::new(py, &data))) {
                            e.print(py);
                        }
                    }
                }
            });
        }
    }

    pub fn create_lobby(&mut self, lobby_type: u32, max_members: u32, cb_on_created: Py<PyAny>) {
        if let Some((client, _)) = &self.client {
            let matchmaking = client.matchmaking();

            let lobby_kind = match lobby_type {
                0 => LobbyType::Private,
                1 => LobbyType::FriendsOnly,
                2 => LobbyType::Public,
                3 => LobbyType::Invisible,
                _ => LobbyType::Private,
            };

            // The GameIdentity tag itself is set from Python right after
            // a successful creation (see on_lobby_created in network_
            // manager.py), not here - Matchmaking<ClientManager> wraps a
            // raw ISteamMatchmaking pointer, which isn't Send, so it
            // can't be captured into this callback (create_lobby
            // requires F: Send, since Steam's own callback dispatch
            // doesn't guarantee which thread invokes it). Tagging
            // synchronously in Python's own on_lobby_created handler,
            // immediately after this call1 hands the new lobby id back,
            // is both simpler and just as immediate in practice.
            matchmaking.create_lobby(lobby_kind, max_members, move |result| {
                Python::with_gil(|py| {
                    let call = match result {
                        Ok(lobby_id) => cb_on_created.call1(py, (lobby_id.raw(),)),
                        Err(err) => cb_on_created.call1(py, (py.None(), err.to_string())),
                    };
                    if let Err(e) = call {
                        e.print(py);
                    }
                });
            });
        }
    }

    pub fn join_lobby(&mut self, lobby_id: u64, cb_on_joined: Py<PyAny>) {
        if let Some((client, _)) = &self.client {
            let matchmaking = client.matchmaking();
            matchmaking.join_lobby(LobbyId::from_raw(lobby_id), move |result| {
                Python::with_gil(|py| {
                    let call = match result {
                        Ok(lobby_id) => cb_on_joined.call1(py, (lobby_id.raw(), py.None())),
                        Err(e) => cb_on_joined.call1(
                            py,
                            (
                                py.None(),
                                PyRuntimeError::new_err(format!("No Lobby Found: {:?}", e)),
                            ),
                        ),
                    };
                    if let Err(e) = call {
                        e.print(py);
                    }
                });
            });
        }
    }

    pub fn leave_lobby(&mut self, lobby_id: u64) {
        if let Some((client, _)) = &self.client {
            let matchmaking = client.matchmaking();
            matchmaking.leave_lobby(LobbyId::from_raw(lobby_id));
        }
    }

    pub fn get_lobby_members(&self, py: Python<'_>, lobby_id: u64) -> PyResult<PyObject> {
        if let Some(client) = &self.client {
            let matchmaking = client.0.matchmaking();
            let lobby = LobbyId::from_raw(lobby_id);
            let members = matchmaking.lobby_members(lobby);

            if members.is_empty() {
                return Err(PyRuntimeError::new_err("Lobby not found or no members"));
            }

            let member_ids: Vec<u64> = members.iter().map(|id| id.raw()).collect();

            let py_list = PyList::new(py, member_ids.clone())?;
            Ok(py_list.into_pyobject(py)?.into())
        } else {
            Err(PyRuntimeError::new_err("Client not initialized"))
        }
    }

    pub fn set_lobby_data(&self, lobby_id: u64, key: &str, value: &str) -> PyResult<bool> {
        if let Some((client, _)) = &self.client {
            let matchmaking = client.matchmaking();
            Ok(matchmaking.set_lobby_data(LobbyId::from_raw(lobby_id), key, value))
        } else {
            Err(PyRuntimeError::new_err("Client not initialized"))
        }
    }

    /// Reads one lobby data key (see set_lobby_data). Returns None if the
    /// key isn't set. Works on any lobby whose data this client has cached -
    /// i.e. every lobby a get_lobby_list result just returned, or one it's in.
    pub fn get_lobby_data(&self, lobby_id: u64, key: &str) -> PyResult<Option<String>> {
        if let Some((client, _)) = &self.client {
            let matchmaking = client.matchmaking();
            Ok(matchmaking
                .lobby_data(LobbyId::from_raw(lobby_id), key)
                .map(|s| s.to_string()))
        } else {
            Err(PyRuntimeError::new_err("Client not initialized"))
        }
    }

    pub fn get_lobby_list(&mut self, cb_on_list: Py<PyAny>) {
        if let Some((client, _)) = &self.client {
            let matchmaking = client.matchmaking();
            // Steam-side filter (AddRequestLobbyListStringFilter),
            // applied before RequestLobbyList - Steam itself only
            // returns lobbies carrying this exact tag, rather than this
            // binding fetching every public lobby on the AppID and
            // filtering client-side (which would need a get_lobby_data
            // round trip per lobby just to find out most of them aren't
            // even this game). Every lobby create_lobby produces is
            // already tagged with the same key/value (see its own
            // comment), so this is the read-side half of that pairing.
            // Confirmed via a temporary unfiltered-search diagnostic
            // that RequestLobbyList itself works fine and this exact
            // string filter was the actual problem - see this crate's
            // own [patch.crates-io] comment in Cargo.toml for the real
            // bug (steamworks 0.11.0's own filter functions never
            // null-terminated the strings they handed to Steam's C
            // API) and vendor/steamworks-0.11.0 for the fix.
            matchmaking.add_request_lobby_list_string_filter(StringFilter(
                LobbyKey::new(GAME_IDENTITY_KEY),
                GAME_IDENTITY_VALUE,
                StringFilterKind::Include,
            ));
            // Without this, RequestLobbyList silently uses
            // ELobbyDistanceFilterDefault - Valve's own docs describe
            // this as restricted to "the same immediate region" as the
            // searching client, NOT worldwide. Confirmed as the actual
            // cause of one real machine hosting a correctly-tagged
            // lobby that a second, geographically different machine's
            // search still couldn't find at all - nothing wrong with
            // the tag or the filter above, Steam was just never
            // returning lobbies outside the searcher's own region in
            // the first place. This game has no reason to restrict
            // matchmaking by geography at all, so search worldwide.
            matchmaking.set_request_lobby_list_distance_filter(DistanceFilter::Worldwide);
            matchmaking.request_lobby_list(move |result| {
                Python::with_gil(|py| {
                    // Every call1 in this file used to be `let _ =
                    // call1(...)`, silently discarding whatever
                    // exception the Python callback raised - which is
                    // exactly what hid a real bug for a while: Python's
                    // own handle_lobby_list calling back into this same
                    // PySteamClient instance's create_lobby/join_lobby
                    // (both &mut self) from INSIDE this callback (itself
                    // running inside run_callbacks' &self borrow) raised
                    // PyO3's "RuntimeError: Already borrowed" on every
                    // single call, with nothing anywhere to show for it.
                    // e.print(py) at least gets it into stderr/the
                    // console instead of vanishing.
                    let call = match result {
                        Ok(lobbies) => {
                            let ids: Vec<u64> = lobbies.iter().map(|id| id.raw()).collect();
                            cb_on_list.call1(py, (ids, py.None()))
                        }
                        Err(e) => cb_on_list.call1(py, (Vec::<u64>::new(), e.to_string())),
                    };
                    if let Err(e) = call {
                        e.print(py);
                    }
                });
            });
        }
    }

    pub fn set_lobby_changed_callback(&mut self, cb: Py<PyAny>) {
        let mut guard = self.cb_lobby_changed.lock().unwrap();
        *guard = Some(cb);
    }

    pub fn set_connection_failed_callback(&mut self, cb: Py<PyAny>) {
        let mut guard = self.cb_conn_failed.lock().unwrap();
        *guard = Some(cb);
    }

    pub fn set_message_recv_callback(&mut self, cb: Py<PyAny>) {
        let mut guard = self.cb_message_recv.lock().unwrap();
        *guard = Some(cb);
    }

    pub fn send_message_to(
        &self,
        steam_id: u64,
        message_type: i32,
        channel: u32,
        message: &[u8],
    ) {
        if let Some(networking) = &self.messages {
            let flags = SendFlags::from_bits(message_type).unwrap_or(SendFlags::RELIABLE);
            // Only a FAILED send is worth printing (and only the first few
            // per peer, so a dead session can't flood the terminal).
            if let Err(e) = networking.send_message_to_user(
                NetworkingIdentity::new_steam_id(SteamId::from_raw(steam_id)),
                flags,
                message,
                channel,
            ) {
                let mut counts = self.send_errors.lock().unwrap();
                let n = counts.entry(steam_id).or_insert(0u32);
                *n += 1;
                if *n <= 5 {
                    eprintln!("[py_steam_net] send to {} failed: {:?} (flags {})", steam_id, e, message_type);
                }
            }
        }
    }

    pub fn own_steam_id(&self) -> u64 {
        if let Some((client, _)) = &self.client {
            client.user().steam_id().raw()
        } else {
            0
        }
    }
}